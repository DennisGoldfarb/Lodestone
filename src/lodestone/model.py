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
        # Project the sequence representation into a run-agnostic feature space
        self.feature_head = nn.Linear(d_model, run_dim)
        # Global bias predicting split-normal parameters independent of run
        self.bias_head = nn.Linear(run_dim, 3)
        # Run-specific weights producing dataset dependent adjustments
        self.k_factor_head = nn.Embedding(num_runs, run_dim * 3)
        self.num_charge = num_charge
        # Precompute charge states for constructing the distribution
        self.register_buffer("charges", torch.arange(num_charge).float())

    def _split_normal_logits(
        self, mu: torch.Tensor, sigma_l: torch.Tensor, sigma_r: torch.Tensor
    ) -> torch.Tensor:
        """Return unnormalized log-probabilities for discrete charge states.

        Parameters are the mode ``mu`` and the left/right spreads ``sigma_l`` and
        ``sigma_r`` of a split normal distribution.  The resulting logits are
        shaped ``[batch, num_charge]`` so they can be compared directly against
        target distributions without unwanted broadcasting.
        """

        charges = self.charges  # [num_charge]
        diff = charges.unsqueeze(0) - mu.unsqueeze(-1)
        left = charges.unsqueeze(0) <= mu.unsqueeze(-1)
        sigma = torch.where(left, sigma_l.unsqueeze(-1), sigma_r.unsqueeze(-1))
        # Negative squared distance scaled by variance gives unnormalized logits
        logits = -0.5 * (diff**2) / (sigma**2)
        return logits

    def forward(
        self, x: torch.Tensor, run_ids: torch.Tensor, return_bias: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # x: [B, L, V]
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        feats = self.feature_head(x)
        params_bias = self.bias_head(feats)
        k_weights = self.k_factor_head(run_ids).view(run_ids.size(0), 3, feats.size(1))
        k_params = torch.bmm(k_weights, feats.unsqueeze(-1)).squeeze(-1)
        params_full = params_bias + k_params

        def params_to_logits(params: torch.Tensor) -> torch.Tensor:
            mu, sigma_l, sigma_r = params.chunk(3, dim=-1)
            mu = torch.sigmoid(mu.squeeze(-1)) * (self.num_charge - 1)
            sigma_l = F.softplus(sigma_l.squeeze(-1)) + 1e-6
            sigma_r = F.softplus(sigma_r.squeeze(-1)) + 1e-6
            return self._split_normal_logits(mu, sigma_l, sigma_r)

        logits_full = params_to_logits(params_full)
        if return_bias:
            logits_bias = params_to_logits(params_bias)
            return logits_bias, logits_full
        return logits_full


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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        x, y, run_ids, mask = batch
        preds_bias, preds_full = self(x, run_ids, return_bias=True)
        bias_loss_all = F.mse_loss(
            torch.softmax(preds_bias, dim=-1), y, reduction="none"
        )
        full_loss_all = F.mse_loss(
            torch.softmax(preds_full, dim=-1), y, reduction="none"
        )
        mask = mask.float()
        bias_loss = (bias_loss_all * mask).sum() / mask.sum()
        full_loss = (full_loss_all * mask).sum() / mask.sum()
        loss = 0.9 * bias_loss + 0.1 * full_loss
        self.log("train_loss_epoch", full_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_bias_loss_epoch", bias_loss, on_step=False, on_epoch=True, prog_bar=False)
        if (self.global_step + 1) % 50 == 0:
            self.log("train_loss", full_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train_bias_loss", bias_loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        x, y, run_ids, mask = batch
        preds_bias, preds_full = self(x, run_ids, return_bias=True)
        bias_loss_all = F.mse_loss(
            torch.softmax(preds_bias, dim=-1), y, reduction="none"
        )
        full_loss_all = F.mse_loss(
            torch.softmax(preds_full, dim=-1), y, reduction="none"
        )
        mask = mask.float()
        bias_loss = (bias_loss_all * mask).sum() / mask.sum()
        full_loss = (full_loss_all * mask).sum() / mask.sum()
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

        # Scatter plot of the first two dimensions of the run-specific k-factor weights
        run_weights = self.model.k_factor_head.weight.view(self.hparams.num_runs, 3, -1)
        if run_weights.size(2) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(run_weights[:, 0, 0].cpu(), run_weights[:, 0, 1].cpu())
            ax.set_xlabel("k_factor_dim_0")
            ax.set_ylabel("k_factor_dim_1")
            ax.set_title("Run k-factor weights")
            self.logger.experiment.log({"run_dim_scatter": wandb.Image(fig)}, commit=False)
            plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
