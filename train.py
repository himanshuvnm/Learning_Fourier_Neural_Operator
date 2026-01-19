import os
import argparse
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.io import loadmat


# ----------------------------
# Model 
# ----------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)  # (B, Cin, N) -> (B, Cin, N//2+1)
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes: int, width: int):
        super().__init__()
        self.modes1 = modes
        self.width = width

        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)          # (B, N, 2) -> (B, N, width)
        x = x.permute(0, 2, 1)   # (B, width, N)

        x = F.relu(self.conv0(x) + self.w0(x))
        x = F.relu(self.conv1(x) + self.w1(x))
        x = F.relu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 1)   # (B, N, width)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)          # (B, N, 1)
        return x.squeeze(-1)     # (B, N)


# ----------------------------
# Data utilities
# ----------------------------
def _make_xy_from_mat(
    mat: Dict[str, Any],
    key_output: str,
    t_in: Tuple[int, int],
    t_out: Tuple[int, int],
    sub: int,
    n_total: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matches your notebook logic:
      x_data = output[:, 8:16, :].reshape(N, 1024*8)[:, ::sub]
      y_data = output[:,16:24, :].reshape(N, 1024*8)[:, ::sub]
    """
    if key_output not in mat:
        raise KeyError(f"'{key_output}' not found in .mat keys: {list(mat.keys())}")

    out = mat[key_output]  # expected shape: (N, T, X)
    if out.ndim != 3:
        raise ValueError(f"Expected output to be rank-3 (N,T,X), got shape {out.shape}")

    if out.shape[0] < n_total:
        raise ValueError(f"n_total={n_total} but dataset has only N={out.shape[0]} trajectories")

    out = out[:n_total]

    x_np = out[:, t_in[0] : t_in[1], :].reshape(n_total, -1)[:, ::sub]
    y_np = out[:, t_out[0] : t_out[1], :].reshape(n_total, -1)[:, ::sub]

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    return x, y


def _add_grid_channel(x: torch.Tensor, resolution_original: int, sub: int) -> torch.Tensor:
    """
    Builds a grid in [0,1] at original resolution, downsamples with same stride,
    and concatenates as second channel: (value, grid).
    """
    grid_all = np.linspace(0, 1, resolution_original).reshape(resolution_original, 1).astype(np.float64)
    grid = torch.tensor(grid_all[::sub, :], dtype=torch.float32)  # (Ndown, 1)

    # x: (B, Ndown)
    x = x.reshape(x.shape[0], -1, 1)  # (B, Ndown, 1)
    grid = grid.repeat(x.shape[0], 1, 1)  # (B, Ndown, 1)
    return torch.cat([x, grid], dim=2)  # (B, Ndown, 2)


class BurgersDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage=None):
        dcfg = self.cfg["data"]
        mat = loadmat(dcfg["dataset_path"])

        x, y = _make_xy_from_mat(
            mat=mat,
            key_output=dcfg["key_output"],
            t_in=(dcfg["t_in"]["start"], dcfg["t_in"]["end"]),
            t_out=(dcfg["t_out"]["start"], dcfg["t_out"]["end"]),
            sub=dcfg["sub"],
            n_total=dcfg["n_total"],
        )

        num_train = dcfg["num_train"]
        num_test = dcfg["num_test"]

        x_train, y_train = x[:num_train], y[:num_train]
        x_test, y_test = x[-num_test:], y[-num_test:]

        x_train = _add_grid_channel(x_train, dcfg["resolution_original"], dcfg["sub"])
        x_test = _add_grid_channel(x_test, dcfg["resolution_original"], dcfg["sub"])

        bsz = self.cfg["train"]["batch_size"]
        self.train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=bsz, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=bsz, shuffle=False)
        self.test_loader = self.val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


# ----------------------------
# Lightning module
# ----------------------------
class LitFNO(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]
        self.model = FNO1d(modes=mcfg["modes"], width=mcfg["width"])

    def forward(self, x):
        return self.model(x)

    def _loss(self, y_hat, y):
        return F.mse_loss(y_hat.view(-1, 1), y.view(-1, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        tcfg = self.cfg["train"]
        opt = torch.optim.Adam(
            self.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=tcfg["lr_step_size"], gamma=tcfg["lr_gamma"])
        return {"optimizer": opt, "lr_scheduler": sch}


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/burgers_re10.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # dirs
    out_dir = os.path.join(cfg["logging"]["results_dir"], cfg["logging"]["run_name"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    dm = BurgersDataModule(cfg)
    model = LitFNO(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg["logging"]["monitor"],
        mode=cfg["logging"]["mode"],
        dirpath=ckpt_dir,
        filename="fno-{epoch:02d}-{val_loss:.6f}",
        save_top_k=cfg["logging"]["save_top_k"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["train"]["max_epochs"],
        accelerator=cfg["runtime"]["accelerator"],
        devices=cfg["runtime"]["devices"],
        precision=cfg["runtime"]["precision"],
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        default_root_dir=out_dir,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)
    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
