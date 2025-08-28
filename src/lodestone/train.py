import json
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from .data import PeptideDataModule
from .model import LodestoneLightningModule

os.environ["WANDB_DIR"] = "/scratch1/fs1/d.goldfarb/Lodestone/"
os.environ["WANDB_CACHE_DIR"] = "/scratch1/fs1/d.goldfarb/Lodestone/"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    logger = WandbLogger(project="lodestone", log_model=False)

    datamodule = PeptideDataModule(
        cfg["csv_path"],
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 0),
    )
    datamodule.setup("fit")

    model = LodestoneLightningModule(
        num_runs=len(datamodule.run_mapping),
        lr=cfg.get("lr", 1e-3),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        run_dim=cfg.get("run_dim", 32),
    )

    trainer_args = {"max_epochs": cfg.get("max_epochs", 10), "logger": logger}
    for key in ["limit_train_batches", "limit_val_batches", "fast_dev_run"]:
        if key in cfg:
            trainer_args[key] = cfg[key]
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
