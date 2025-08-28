from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .data import VOCAB


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class LodestoneModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, run_dim: int, num_runs: int, num_charge: int = 5):
        super().__init__()
        self.embed = nn.Linear(len(VOCAB), d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.weight_head = nn.Linear(d_model, run_dim * num_charge)
        self.run_params = nn.Embedding(num_runs, run_dim)
        self.k_factors = nn.Embedding(num_runs, num_charge)
        # Global bias applied to k-factor head to allow predictions without
        # any dataset specific knowledge.
        self.k_factor_bias = nn.Parameter(torch.ones(num_charge))

    def forward(
        self, x: torch.Tensor, run_ids: torch.Tensor, return_bias: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # x: [B, L, V]
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        weights = self.weight_head(x).view(x.size(0), -1, self.run_params.embedding_dim)
        run_vec = self.run_params(run_ids).unsqueeze(-1)
        preds = torch.bmm(weights, run_vec).squeeze(-1)
        k_bias = self.k_factor_bias.unsqueeze(0)
        k_coeff = self.k_factors(run_ids)
        preds_full = preds * (k_bias + k_coeff)
        if return_bias:
            preds_bias = preds * k_bias
            return preds_bias, preds_full
        return preds_full


class LodestoneLightningModule(pl.LightningModule):
    def __init__(self, num_runs: int, lr: float = 1e-3, d_model: int = 128, nhead: int = 4, num_layers: int = 2, run_dim: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.model = LodestoneModel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            run_dim=run_dim,
            num_runs=num_runs,
        )
        self.val_examples = []

    def forward(
        self, x: torch.Tensor, run_ids: torch.Tensor, return_bias: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        return self.model(x, run_ids, return_bias=return_bias)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        x, y, run_ids = batch
        preds_bias, preds_full = self(x, run_ids, return_bias=True)
        bias_loss = F.mse_loss(torch.softmax(preds_bias, dim=-1), y)
        full_loss = F.mse_loss(torch.softmax(preds_full, dim=-1), y)
        loss = 0.9 * bias_loss + 0.1 * full_loss
        self.log("train_loss_epoch", full_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_bias_loss_epoch", bias_loss, on_step=False, on_epoch=True, prog_bar=False)
        if (self.global_step + 1) % 50 == 0:
            self.log("train_loss", full_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train_bias_loss", bias_loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        x, y, run_ids = batch
        preds_bias, preds_full = self(x, run_ids, return_bias=True)
        bias_loss = F.mse_loss(torch.softmax(preds_bias, dim=-1), y)
        full_loss = F.mse_loss(torch.softmax(preds_full, dim=-1), y)
        self.log("val_loss", full_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_bias_loss", bias_loss, on_step=False, on_epoch=True, prog_bar=False)
        if len(self.val_examples) < 10:
            self.val_examples.append(
                (
                    y[0].detach().cpu(),
                    torch.softmax(preds_full.detach(), dim=-1)[0].detach().cpu(),
                )
            )
        return full_loss

    def on_validation_epoch_end(self):
        import matplotlib.pyplot as plt
        import wandb

        for y, p in self.val_examples:
            if (y > 0).sum() >= 2:
                charges = range(y.size(-1))
                fig, ax = plt.subplots()
                ax.bar(charges, y.numpy(), color="blue")
                ax.bar(charges, -p.numpy(), color="orange")
                ax.set_xlabel("Charge")
                ax.set_ylabel("Abundance")
                ax.set_title("Observed (top) vs Predicted (bottom)")
                self.logger.experiment.log({"mirror_plot": wandb.Image(fig)}, commit=False)
                plt.close(fig)
                break
        self.val_examples = []

        # Scatter plot of the first two run_dim weights for each dataset
        run_weights = self.model.run_params.weight.detach().cpu()
        if run_weights.size(1) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(run_weights[:, 0], run_weights[:, 1])
            ax.set_xlabel("run_dim_0")
            ax.set_ylabel("run_dim_1")
            ax.set_title("Run parameter weights")
            self.logger.experiment.log({"run_dim_scatter": wandb.Image(fig)}, commit=False)
            plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
