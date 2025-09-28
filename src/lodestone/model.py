from pathlib import Path
from typing import Tuple, List

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
        # Precompute charge states for constructing the distribution. Charge
        # states start at 1 rather than 0 because the training data does not
        # contain a zero-charge precursor.
        self.register_buffer("charges", torch.arange(1, num_charge + 1).float())

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
        padding_mask = (x.abs().sum(dim=-1) == 0)
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        non_pad = (~padding_mask).unsqueeze(-1).to(x.dtype)
        lengths = non_pad.sum(dim=1).clamp(min=1.0)
        x = (x * non_pad).sum(dim=1) / lengths
        feats = self.feature_head(x)
        params_bias = self.bias_head(feats)
        k_weights = self.k_factor_head(run_ids).view(run_ids.size(0), 3, feats.size(1))
        k_params = torch.bmm(k_weights, feats.unsqueeze(-1)).squeeze(-1)
        params_full = params_bias + k_params

        def params_to_logits(params: torch.Tensor) -> torch.Tensor:
            mu, sigma_l, sigma_r = params.chunk(3, dim=-1)
            # Map the unconstrained mode to the [1, num_charge] interval so that
            # it aligns with the enumerated charge states above.
            mu = torch.sigmoid(mu.squeeze(-1)) * (self.num_charge - 1) + 1
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
        self.val_examples = {}
        self.selected_sequence = None
        self.run_id_to_name: dict[int, str] = {}

    def forward(
        self, x: torch.Tensor, run_ids: torch.Tensor, return_bias: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        return self.model(x, run_ids, return_bias=return_bias)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
    ):
        x, y, run_ids, mask, _ = batch
        preds_bias, preds_full = self(x, run_ids, return_bias=True)
        bias_loss_all = -(y * F.log_softmax(preds_bias, dim=-1))
        full_loss_all = -(y * F.log_softmax(preds_full, dim=-1))
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
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
    ):
        x, y, run_ids, mask, seqs = batch
        preds_bias, preds_full = self(x, run_ids, return_bias=True)
        bias_loss_all = -(y * F.log_softmax(preds_bias, dim=-1))
        full_loss_all = -(y * F.log_softmax(preds_full, dim=-1))
        mask = mask.float()
        bias_loss = (bias_loss_all * mask).sum() / mask.sum()
        full_loss = (full_loss_all * mask).sum() / mask.sum()
        self.log("val_loss", full_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_bias_loss", bias_loss, on_step=False, on_epoch=True, prog_bar=False)

        if not self.run_id_to_name:
            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is not None and hasattr(datamodule, "run_mapping"):
                self.run_id_to_name = {idx: name for name, idx in datamodule.run_mapping.items()}

        softmax_bias = torch.softmax(preds_bias.detach(), dim=-1).detach().cpu()
        softmax_full = torch.softmax(preds_full.detach(), dim=-1).detach().cpu()
        mask_cpu = mask.detach().cpu()

        for i, seq in enumerate(seqs):
            run_id = int(run_ids[i].item())
            dataset_name = self.run_id_to_name.get(run_id, str(run_id))
            mask_sum = mask[i].sum()
            if mask_sum > 0:
                example_bias_loss = (bias_loss_all[i] * mask[i]).sum() / mask_sum
                example_full_loss = (full_loss_all[i] * mask[i]).sum() / mask_sum
            else:
                example_bias_loss = bias_loss_all[i].mean()
                example_full_loss = full_loss_all[i].mean()

            entry = self.val_examples.setdefault(seq, {})
            if dataset_name not in entry:
                entry[dataset_name] = (
                    y[i].detach().cpu(),
                    softmax_bias[i],
                    softmax_full[i],
                    float(example_bias_loss.detach()),
                    float(example_full_loss.detach()),
                    mask_cpu[i],
                )
                if (
                    self.selected_sequence is None
                    and self._sequence_is_shared_multi_charge(entry)
                ):
                    self.selected_sequence = seq
        return full_loss

    def on_validation_epoch_end(self):
        import matplotlib.pyplot as plt
        import wandb

        if self.selected_sequence is not None:
            seq = self.selected_sequence
            dataset_entries = self.val_examples.get(seq, {})
            if dataset_entries:
                dataset_names = sorted(dataset_entries.keys())
                num_datasets = len(dataset_names)
                fig, axes = plt.subplots(
                    num_datasets,
                    2,
                    figsize=(10, 4 * num_datasets),
                    sharey=True,
                    squeeze=False,
                )
                charges = range(1, next(iter(dataset_entries.values()))[0].size(-1) + 1)
                for row_idx, dataset_name in enumerate(dataset_names):
                    y, p_bias, p_full, bias_loss, full_loss, mask = dataset_entries[dataset_name]
                    panels = [
                        ("Bias only", p_bias, bias_loss),
                        ("Run adjusted", p_full, full_loss),
                    ]
                    for col_idx, (title, pred, loss) in enumerate(panels):
                        ax = axes[row_idx, col_idx]
                        ax.bar(charges, y.numpy(), color="blue")
                        ax.bar(charges, -pred.numpy(), color="orange")
                        ax.set_xlabel("Charge")
                        ax.set_title(f"{dataset_name} — {title}\nloss={loss:.4f}")
                        if mask is not None:
                            masked = mask.numpy().astype(bool)
                            for charge_idx, valid in enumerate(masked, start=1):
                                if not valid:
                                    ax.axvspan(charge_idx - 0.5, charge_idx + 0.5, color="gray", alpha=0.2)
                    axes[row_idx, 0].set_ylabel("Abundance")
                fig.suptitle(f"Sequence present in multiple datasets: {seq}")
                output_dir = Path("/storage1/fs1/d.goldfarb/Active/Projects/Lodestone")
                output_dir.mkdir(parents=True, exist_ok=True)
                epoch = getattr(self.trainer, "current_epoch", self.current_epoch)
                fig.savefig(
                    output_dir / f"validation_mirror_plot_epoch_{epoch:04d}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
                self.logger.experiment.log({"mirror_plot": wandb.Image(fig)}, commit=False)
                plt.close(fig)

        self.val_examples = {}
        self.selected_sequence = None

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

    def _sequence_is_shared_multi_charge(self, dataset_entries: dict[str, tuple]):
        """Return True when sequence spans multiple datasets and is multi-charge.

        The user wants the validation mirror plot peptide to satisfy three
        conditions:

        1. It must be observed in at least two distinct datasets.
        2. In at least one of those datasets the peptide should have evidence for
           two or more charge states.
        3. Within that dataset at least two observed charges must individually
           account for more than 25% of the total intensity while also being
           charge state 2 or higher.

        The mask tracks which charge states are valid for the example.  When a
        mask is unavailable we fall back to the target distribution ``y`` to
        approximate the number of observed charge states as well as the
        intensity contribution per charge.
        """

        if len(dataset_entries) < 2:
            return False

        for entry in dataset_entries.values():
            if len(entry) < 6:
                continue
            y, _, _, _, _, mask = entry

            y_tensor = torch.as_tensor(y)
            mask_tensor = None
            if mask is not None:
                mask_tensor = torch.as_tensor(mask)
                if mask_tensor.numel() == 0:
                    mask_tensor = None

            charge_states = torch.arange(1, y_tensor.numel() + 1)

            if mask_tensor is not None:
                observed_mask = mask_tensor > 0.5
            else:
                observed_mask = y_tensor > 0

            strong_mask = y_tensor > 0.25
            meets_charge_threshold = charge_states >= 2
            qualifying = observed_mask & strong_mask & meets_charge_threshold

            if qualifying.sum().item() >= 2:
                return True

        return False
