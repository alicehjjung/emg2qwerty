from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

import torch
torch.backends.cudnn.enabled = False

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        pe = self._generate_pe(d_model, max_len)
        self.register_buffer('pe', pe)

    def _generate_pe(self, d_model: int, length: int) -> torch.Tensor:
        pe = torch.zeros(length, 1, d_model)
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) 
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        x = x * math.sqrt(self.d_model)
        
        if T > self.pe.size(0):
            new_pe = self._generate_pe(self.d_model, T).to(x.device)
            x = x + new_pe
        else:
            x = x + self.pe[:T]
            
        return x
    
class TransformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Frontend
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T, N, num_features)
        )

        # Project frontend features to transformer hidden size
        self.input_proj = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # input shape: (T, N, D)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _make_padding_mask(
        self, input_lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        # Returns shape (N, T), where True means padded position
        positions = torch.arange(max_len, device=input_lengths.device).unsqueeze(0)
        return positions >= input_lengths.unsqueeze(1)

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.frontend(inputs)          # (T, N, num_features)
        x = self.input_proj(x)             # (T, N, d_model)
        x = self.pos_encoder(x)            # (T, N, d_model)

        padding_mask = self._make_padding_mask(
            input_lengths=input_lengths,
            max_len=x.size(0),
        )  # (N, T)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )  # (T, N, d_model)

        emissions = self.classifier(x)     # (T, N, num_classes)

        # Transformer encoder does not change time length
        emission_lengths = input_lengths
        return emissions, emission_lengths

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions, emission_lengths = self.forward(inputs, input_lengths)

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()

        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )