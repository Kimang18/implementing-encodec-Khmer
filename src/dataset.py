import warnings
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

warnings.filterwarnings("ignore")


@dataclass
class AudioDataset:
    segment_length: int = 16000
    sample_rate: int = 16000

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = []
        for f in features:
            waveform = f["audio"]["array"]
            # NOTE: make sure that waveform is mono, and sampling_rate is 16000
            # not do it here
            n_samples = waveform.shape[0]
            if n_samples > self.segment_length:
                # Random start position
                start = torch.randint(
                    0, n_samples - self.segment_length - 1, (1,)
                ).item()
                waveform = waveform[start : start + self.segment_length]
            elif n_samples < self.segment_length:
                # pad if audio is shorter than segment_length
                padding = self.segment_length - n_samples
                waveform = F.pad(waveform, (0, padding), mode="constant", value=0)

            waveforms.append(
                torch.as_tensor(waveform[None, :])
            )  # add channel dimension because the dataset used is mono

        return {"waveforms": torch.stack(waveforms)}


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        sample_rate=16000,
        hf_repo="Digital-Divide-Data/khmer-speech-dataset",
        segment_length=16000,
        bs_per_device=4,
        n_worker_per_device=0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        ds = load_dataset(self.hparams.hf_repo, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.hparams.sample_rate))
        self.ds_train = ds["train"]
        self.ds_validation = ds["test"]

    def get_dataloader(self, dataset, batch_size, num_workers, shard_on_rank=True):
        def worker_init_fn(worker_id: int):
            worker_info = get_worker_info()
            if worker_info is not None:
                worker_ds = worker_info.dataset
                worker_ds.hf_ds = worker_ds.hf_ds.shard(
                    num_shards=worker_info.num_workers, index=worker_info.id
                )

        class PytorchHFIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, hf_ds):
                self.hf_ds = hf_ds

            def __iter__(self):
                while True:
                    yield from self.hf_ds

            def state_dict(self):
                return self.hf_ds.state_dict()

            def load_state_dict(self, state_dict):
                self.hf_ds.load_state_dict(state_dict)

        if shard_on_rank:
            dataset = split_dataset_by_node(
                dataset,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            )
            dataset = PytorchHFIterableDataset(dataset)

        return StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=AudioDataset(
                self.hparams.segment_length, self.hparams.sample_rate
            ),
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def train_dataloader(self):
        loader = self.get_dataloader(
            self.ds_train,
            self.hparams.bs_per_device,
            self.hparams.n_worker_per_device,
            True,
        )
        if self.trainer is not None and hasattr(
            self.trainer.lightning_module, "_restore_dataset"
        ):
            if self.trainer.lightning_module._restore_dataset:
                loader.load_state_dict(self.trainer.lightning_module._restore_dataset)
                self.trainer.lightning_module._restore_dataset = None
        return loader

    def val_dataloader(self):
        return self.get_dataloader(
            self.ds_validation, self.hparams.bs_per_device, 0, False
        )
