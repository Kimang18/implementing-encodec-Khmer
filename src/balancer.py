from collections import defaultdict

import torch
import torch.distributed as dist


def average_metrics_across_gpus(metrics, count=1):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        keys, values = zip(*metrics.items())
        device = values[0].device
        # tensor = [v1, v2, ..., count]
        tensor = torch.tensor(
            list(values) + [count], dtype=torch.float32, device=device
        )

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
        return dict(zip(keys, averaged))
    return metrics


def averager(beta=1.0, state=None):
    if state is not None:
        fix = defaultdict(float, state.get("fix", {}))
        total = defaultdict(float, state.get("total", {}))
    else:
        fix = defaultdict(float)
        total = defaultdict(float)

    def _update(metrics):
        # metrics = {"name": norm, "name2": norm2, ...}
        nonlocal total, fix

        for key, value in metrics.items():
            if value is not None:
                total[key] = total[key] * beta + float(value)
                fix[key] = fix[key] * beta + 1
        return {
            key: (tot / fix[key] if tot is not None else None)
            for key, tot in total.items()
        }

    def get_state():
        return {
            "fix": dict(fix),
            "total": dict(total),
        }

    def load_state(state):
        nonlocal fix, total
        fix = defaultdict(float, state.get("fix", {}))
        total = defaultdict(float, state.get("total", {}))

    _update.get_state = get_state
    _update.load_state = load_state
    return _update


class Balancer:
    def __init__(
        self,
        weights,
        total_norm=1,
        ema_decay=0.999,
        eps=1e-12,
        max_scale=100,
        min_scale=0.01,
    ):
        self.weights = weights
        self.total_norm = total_norm
        self.eps = eps
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.averager = averager(ema_decay)

    def backward(self, losses, input, retain_graph=False):
        norms = {}
        grads = {}

        for name, loss in losses.items():
            grad, *_ = torch.autograd.grad(
                loss, [input], retain_graph=True, allow_unused=True
            )

            if grad is not None:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims)  # (batch, )

                norm = norm.mean()  # (1, )

                norms[name] = norm
                grads[name] = grad

                count = len(grad)
            else:
                norms[name] = None
                grads[name] = None

        avg_norms = average_metrics_across_gpus(self.averager(norms), count)

        total_weights = sum([self.weights[k] for k in avg_norms])
        ratios = {k: w / total_weights for (k, w) in self.weights.items()}

        out_grad = 0

        for name, avg_norm in avg_norms.items():
            scale = ratios[name] * self.total_norm / (self.eps + avg_norm)
            scale = max(self.min_scale, min(scale, self.max_scale))
            if grads[name] is not None:
                grad = grads[name] * scale
                out_grad += grad

        input.backward(out_grad, retain_graph=retain_graph)

    def state_dict(self):
        """save the state dict"""
        return {"averager_state": self.averager.get_state()}

    def load_state(self, state_dict):
        """load the state dict"""
        self.averager.load_state(state_dict["averager_state"])
