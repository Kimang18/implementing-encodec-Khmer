from typing import Any

import hydra
import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import TQDMProgressBar
from omegaconf import DictConfig
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchmetrics import MeanMetric
from transformers import get_cosine_schedule_with_warmup

from src.balancer import Balancer
from src.dataset import AudioDataModule
from src.loss import discriminator_loss, generator_loss
from src.modules.discriminator import Discriminator, DiscriminatorConfig
from src.modules.encodec import EncodecConfig, EncodecModel


class TrainModule(L.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        disc_update_prob=0.66,
        max_grad_norm=0.5,
        time_loss_lambda=0.1,
        frequency_loss_lambda=1.0,
        generator_loss_lambda=3.0,
        feature_loss_lambda=3.0,
        quantizer_loss_lambda=3.0,
        use_balancer=True,
        compile=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        self.discriminator = discriminator
        self.weights = {
            "time_loss": time_loss_lambda,
            "frequency_loss": frequency_loss_lambda,
            "generator_loss": generator_loss_lambda,
            "feature_loss": feature_loss_lambda,
        }
        # self.train_loss = MeanMetric()
        # self.train_components = nn.ModuleDict(
        #     {name: MeanMetric() for name in self.weights}
        # )
        # for loss_name in ["quantizer_loss"]:
        #     self.train_components[loss_name] = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_components = nn.ModuleDict(
            {"time_loss": MeanMetric(), "frequency_loss": MeanMetric()}
        )

        # Important: this property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()

        waveforms = batch["waveforms"]
        bs = len(waveforms)

        # generator step #
        g_opt.zero_grad(set_to_none=True)

        output = self.generator(waveforms)
        _, logits_fake, fmap_real, fmap_fake = self.discriminator(
            waveforms, output["decoded"]
        )

        losses = generator_loss(
            fmap_real,
            fmap_fake,
            logits_fake,
            waveforms,
            output["decoded"],
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
        )
        if not self.hparams.use_balancer:
            total_gen_loss = sum(
                loss * self.weights[name] for name, loss in losses.items()
            )
            total_gen_loss += (
                self.hparams.quantizer_loss_lambda * output["quantizer_loss"]
            )
            # total_gen_loss = (
            #     self.weights["time_loss"] * losses["time_loss"]
            #     + self.weights["frequency_loss"] * losses["frequency_loss"]
            #     + self.weights["generator_loss"] * losses["generator_loss"]
            #     + self.weights["feature_loss"] * losses["feature_loss"]
            #     + self.hparams.quantizer_loss_lambda * output["quantizer_loss"]
            # )
            self.manual_backward(total_gen_loss)
        else:
            self.balancer.backward(losses, output["decoded"], retain_graph=True)
            self.manual_backward(
                self.hparams.quantizer_loss_lambda * output["quantizer_loss"]
            )
        # clip gradient norm
        self.clip_gradients(
            g_opt,
            gradient_clip_val=self.hparams.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            g_opt.step()
            g_sch.step()

        # discriminator step #
        d_opt.zero_grad(set_to_none=True)
        rand = torch.rand(size=(), device=waveforms.device)
        dist.broadcast(rand, src=0)

        if rand < self.hparams.disc_update_prob:
            logits_real, logits_fake, _, _ = self.discriminator(
                waveforms, output["decoded"].detach()
            )
            disc_loss = discriminator_loss(logits_real, logits_fake)
            self.manual_backward(disc_loss)
            # clip gradient norm
            self.clip_gradients(
                d_opt,
                gradient_clip_val=self.hparams.max_grad_norm,
                gradient_clip_algorithm="norm",
            )
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                d_opt.step()
                d_sch.step()
        else:
            disc_loss = "skipped"
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                d_sch.step()

        # for name, loss in losses.items():
        #     self.train_components[name].update(loss, weight=bs)
        # self.train_components["quantizer_loss"].update(output["quantizer_loss"])
        self.log(
            "train/quantizer_loss",
            output["quantizer_loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, batch_size=bs)
        self.log(
            "train/discrim_loss", disc_loss, on_step=True, on_epoch=False, batch_size=bs
        )
        self.log(
            "train/gen_lr",
            g_sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            batch_size=bs,
        )
        self.log(
            "train/disc_lr",
            d_sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            batch_size=bs,
        )

    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        bs = len(batch["waveforms"])
        output = self.generator(batch["waveforms"])
        _, logits_fake, fmap_real, fmap_fake = self.discriminator(
            batch["waveforms"], output["decoded"]
        )

        losses = generator_loss(
            fmap_real,
            fmap_fake,
            logits_fake,
            batch["waveforms"],
            output["decoded"],
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
        )
        self.val_loss.update(
            self.weights["time_loss"] * losses["time_loss"]
            + self.weights["frequency_loss"] * losses["frequency_loss"]
            + self.weights["generator_loss"] * losses["generator_loss"]
            + self.weights["feature_loss"] * losses["feature_loss"]
            + self.hparams.quantizer_loss_lambda * output["quantizer_loss"],
            weight=bs,
        )
        self.log(
            "val/loss_total", self.val_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        for name, metric in self.val_components.items():
            metric.update(losses[name], weight=bs)

        for name, metric in self.val_components.items():
            self.log(f"val/{name}", metric, on_step=False, on_epoch=True)

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net, dynamic=True)

        if self.hparams.use_balancer:
            self.balancer = Balancer(weights=self.weights)

    def on_validation_epoch_end(self):
        # TODO: save a few samples
        pass

    def on_save_checkpoint(self, checkpoint):
        train_dataloader = self.trainer.train_dataloader
        if isinstance(train_dataloader, StatefulDataLoader):
            checkpoint["dataset_state_dict"] = train_dataloader.state_dict()
        if self.hparams.use_balancer:
            checkpoint["balancer_state"] = self.balancer.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self._restore_dataset = checkpoint.get("dataset_state_dict", None)
        if self.hparams.use_balancer:
            self.balancer.load_state(checkpoint["balancer_state"])

    def configure_optimizers(self):
        G_optimizer = torch.optim.Adam(
            [p for p in self.generator.parameters()], lr=self.hparams.learning_rate
        )
        D_optimizer = torch.optim.Adam(
            [p for p in self.discriminator.parameters()], lr=self.hparams.learning_rate
        )
        optimizer_steps = int(
            self.trainer.max_steps / self.trainer.accumulate_grad_batches
        )
        warmup_steps = int(0.015 * optimizer_steps)
        G_scheduler = get_cosine_schedule_with_warmup(
            G_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=optimizer_steps,
        )
        D_scheduler = get_cosine_schedule_with_warmup(
            D_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=optimizer_steps,
        )
        return [G_optimizer, D_optimizer], [G_scheduler, D_scheduler]


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(seed=3407, workers=True)
    dm = AudioDataModule(**cfg.data)

    encodec_cfg = EncodecConfig(**cfg.gen)
    generator = EncodecModel(encodec_cfg)
    discrim_cfg = DiscriminatorConfig(**cfg.disc)
    discriminator = Discriminator(discrim_cfg)

    model = TrainModule(generator, discriminator, **cfg.module)
    trainer = L.Trainer(
        # num_sanity_val_step=0,
        enable_progress_bar=True,
        check_val_every_n_epoch=None,
        callbacks=[TQDMProgressBar(refresh_rate=400)],
        **cfg.trainer,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
