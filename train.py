import logging
import os
from typing import Any

import hydra
import lightning as L
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio.transforms as T
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import TQDMProgressBar
from omegaconf import DictConfig
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchmetrics import MeanMetric
from transformers import get_cosine_schedule_with_warmup

from src.balancer import Balancer
from src.dataset import AudioDataModule
from src.modules.discriminator import Discriminator, DiscriminatorConfig
from src.modules.encodec import EnCodecConfig, EnCodecModel
from src.utils import save_audios

# Set the logging level for httpx to WARNING to hide INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def make_buffers_non_persistent(module: nn.Module):
    """Avoid saving MelSpectrogram buffers in checkpoint"""
    for m in module.modules():
        m._non_persistent_buffers_set.update(m._buffers.keys())


class TrainModule(L.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        sample_rate=16000,
        n_mels=80,
        learning_rate=1e-4,
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

        # different MelSpectrogram converters for generator_loss
        ffts = []
        mel_fmin, mel_fmax = 0.0, 8000.0
        for i in range(5, 12):  # 2**5 -> 2**11
            window_size = 2**i
            hop_size = window_size // 4
            ffts.append(
                T.MelSpectrogram(
                    n_fft=window_size,
                    win_length=window_size,
                    hop_length=hop_size,
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    f_min=mel_fmin,
                    f_max=mel_fmax,
                    center=False,
                    window_fn=torch.hann_window,
                )
            )
            make_buffers_non_persistent(ffts[-1])
        self.ffts = nn.ModuleList(ffts)

        # Important: this property activates manual optimization.
        self.automatic_optimization = False

    def _compute_generator_loss(
        self,
        fmap_real,  # list[list[tensor]] -> K discriminators, L layers each
        fmap_fake,  # list[list[tensor]]
        logits_fake,  # list[tensor] -> K discriminators
        input_wav,
        output_wav,
    ):
        # time domain loss #
        time_loss = F.l1_loss(input_wav, output_wav)

        # freq domain loss #
        frequency_loss = 0.0
        for fft in self.ffts:
            input_mel = fft(input_wav)
            output_mel = fft(output_wav)

            frequency_loss = (
                frequency_loss
                + F.l1_loss(input_mel, output_mel)
                + F.mse_loss(input_mel, output_mel)
            )
        frequency_loss = frequency_loss / len(self.ffts)

        # generation capacity #
        if (
            (fmap_real is not None)
            and (fmap_fake is not None)
            and (logits_fake is not None)
        ):
            K = len(logits_fake)
            L = len(fmap_fake[0])

            generator_loss = 0.0
            for lfake in logits_fake:
                generator_loss = generator_loss + torch.mean(F.relu(1 - lfake))
            generator_loss = generator_loss / K

            feature_loss = 0.0
            for freals, ffakes in zip(fmap_real, fmap_fake):
                for freal, ffake in zip(freals, ffakes):
                    feature_loss = feature_loss + F.l1_loss(freal, ffake) / torch.mean(
                        torch.abs(freal)
                    )
            feature_loss = feature_loss / (K * L)
        else:
            generator_loss = torch.tensor(
                [0.0],
                dtype=output_wav.dtype,
                device=output_wav.device,
                requires_grad=True,
            )
            feature_loss = torch.tensor(
                [0.0],
                dtype=output_wav.dtype,
                device=output_wav.device,
                requires_grad=True,
            )

        return {
            "time_loss": time_loss,
            "frequency_loss": frequency_loss,
            "generator_loss": generator_loss,
            "feature_loss": feature_loss,
        }

    def _compute_discriminator_loss(self, logits_real, logits_fake):
        disc_loss = 0.0
        for lreal, lfake in zip(logits_real, logits_fake):
            disc_loss = (
                disc_loss
                + torch.mean(F.relu(1 - lreal))
                + torch.mean(F.relu(1 + lfake))
            )
        disc_loss = disc_loss / len(logits_real)
        return disc_loss

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

        losses = self._compute_generator_loss(
            fmap_real,
            fmap_fake,
            logits_fake,
            waveforms,
            output["decoded"],
            sample_rate=self.hparams.sample_rate,
            n_mels=self.hparams.n_mels,
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
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(rand, src=0)

        if rand < self.hparams.disc_update_prob:
            logits_real, logits_fake, _, _ = self.discriminator(
                waveforms, output["decoded"].detach()
            )
            disc_loss = self._compute_discriminator_loss(logits_real, logits_fake)
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
            disc_loss = 0.0
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
            self.log(
                f"train/{name}",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=bs,
            )
        self.log(
            "train/discrim_loss",
            disc_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )
        self.log(
            "train/gen_lr",
            g_sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )
        self.log(
            "train/disc_lr",
            d_sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )

    def on_train_start(self):
        # self.train_loss.reset()
        self.val_loss.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        bs = len(batch["waveforms"])
        output = self.generator(batch["waveforms"])
        _, logits_fake, fmap_real, fmap_fake = self.discriminator(
            batch["waveforms"], output["decoded"]
        )

        losses = self._compute_generator_loss(
            fmap_real,
            fmap_fake,
            logits_fake,
            batch["waveforms"],
            output["decoded"],
            sample_rate=self.hparams.sample_rate,
            n_mels=self.hparams.n_mels,
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
        if self.trainer.sanity_checking:
            return
        if not self.trainer.is_global_zero:
            return
        self._save_generated_audio()

    def _save_generated_audio(self):
        batch = next(iter(self.trainer.val_dataloaders))
        gens = (
            self.generator.passthrough(batch["waveforms"].to(self.device))
            .detach()
            .cpu()
        )
        gens = [waveform for waveform in gens]
        path_to_save_dir = os.path.join(
            "./gen_samples",
            f"gen_iter_{self.trainer.fit_loop.epoch_loop.total_batch_idx}",
        )
        os.makedirs(path_to_save_dir, exist_ok=True)
        path_to_saves = [
            os.path.join(path_to_save_dir, file)
            for file in [f"gen_{i}.wav" for i in range(len(gens))]
        ]
        save_audios(gens, path_to_saves, self.hparams.sample_rate)

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
            [p for p in self.generator.parameters()],
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.9),
        )
        D_optimizer = torch.optim.Adam(
            [p for p in self.discriminator.parameters()],
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.9),
        )
        optimizer_steps = int(
            self.trainer.max_steps / self.trainer.accumulate_grad_batches
        )
        warmup_steps = int(0.05 * optimizer_steps)
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
    dm = AudioDataModule(cfg.module.sample_rate, **cfg.data)

    encodec_cfg = EnCodecConfig(**cfg.gen)
    generator = EnCodecModel(encodec_cfg)
    discrim_cfg = DiscriminatorConfig(**cfg.disc)
    discriminator = Discriminator(discrim_cfg)

    model = TrainModule(generator, discriminator, **cfg.module)
    trainer = L.Trainer(
        # num_sanity_val_step=0,
        enable_progress_bar=True,
        check_val_every_n_epoch=None,
        callbacks=[TQDMProgressBar(refresh_rate=50)],
        **cfg.trainer,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
