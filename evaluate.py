import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from train import LitFNO, _make_xy_from_mat, _add_grid_channel


def _evaluate_once(cfg, ckpt_path: str, sub: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"]["accelerator"] in ["cuda", "auto"] else "cpu")

    # Load checkpointed Lightning module
    model = LitFNO.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()
    model.to(device)

    dcfg = cfg["data"]
    mat = loadmat(dcfg["dataset_path"])

    x, y = _make_xy_from_mat(
        mat=mat,
        key_output=dcfg["key_output"],
        t_in=(dcfg["t_in"]["start"], dcfg["t_in"]["end"]),
        t_out=(dcfg["t_out"]["start"], dcfg["t_out"]["end"]),
        sub=sub,
        n_total=dcfg["n_total"],
    )

    # Use test split only
    num_test = dcfg["num_test"]
    x_test, y_test = x[-num_test:], y[-num_test:]

    x_test = _add_grid_channel(x_test, dcfg["resolution_original"], sub)

    # Batch evaluation
    bsz = cfg["train"]["batch_size"]
    losses = []
    with torch.no_grad():
        for i in range(0, x_test.shape[0], bsz):
            xb = x_test[i : i + bsz].to(device)
            yb = y_test[i : i + bsz].to(device)

            yhat = model(xb)
            loss = F.mse_loss(yhat.view(-1, 1), yb.view(-1, 1)).item()
            losses.append(loss)

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/burgers_re10.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint .ckpt")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 256, 1024])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    resolution_original = cfg["data"]["resolution_original"]

    print("\nEvaluating checkpoint:", args.ckpt)
    print("Original flattened resolution:", resolution_original)

    for res in args.resolutions:
        if resolution_original % res != 0:
            print(f"[skip] res={res} does not divide resolution_original={resolution_original}")
            continue
        sub = resolution_original // res
        mse = _evaluate_once(cfg, args.ckpt, sub=sub)
        print(f"Resolution {res:4d} | sub={sub:4d} | test_mse={mse:.6e}")


if __name__ == "__main__":
    main()
