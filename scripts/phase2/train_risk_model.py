"""Fase 2 (paso 3) — Entrenamiento de los modelos de predicción de riesgo.

Entrena un modelo zero-inflated de dos cabezas (ver src/risk_model.py) por cada
valor del factor de descuento γ ∈ {0.8, 0.9, 0.95} sobre el dataset multi-label,
con el MISMO split train/val por episodio (GroupShuffleSplit) para los tres γ.

Para cada γ guarda en <output-dir>/gamma_<γ>/:
  - best_model.pt          (checkpoint con menor pérdida de validación)
  - scaler_tabular.pkl     (StandardScaler de las 19 variables tabulares)
y un resumen de métricas (clasificación y regresión) en metrics.json.

Uso:
    python scripts/phase2/train_risk_model.py
    python scripts/phase2/train_risk_model.py --gammas 0.9 --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.risk_model import (N_OBS, N_TAB, RISK_CFG, FullRiskPredictor,
                            focal_bce, masked_weighted_mse)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_SIZE = 0.15
SPLIT_SEED = 42
ALPHA = 10.0  # peso continuo en sampler y pérdida de magnitud: w = 1 + ALPHA·risk

TRAIN_CFG = dict(
    lr=3e-4, batch_size=256, epochs=50, patience=10,
    weight_decay=5e-4, seed=42, lambda_mag=1.0,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(ROOT / "results" / "phase2" / "dataset"
                                            / "dataset_final_multilabel.npz"))
    p.add_argument("--gammas", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    p.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    p.add_argument("--output-dir", default=str(ROOT / "results" / "phase2" / "risk_models"))
    return p.parse_args()


class RiskDataset(Dataset):
    """Entrega (tab, lidar, [log1p(risk), bin], peso)."""

    def __init__(self, tab, lidar, y_pair, weights):
        self.tab = torch.tensor(tab, dtype=torch.float32)
        self.lidar = torch.tensor(lidar, dtype=torch.float32)
        self.y = torch.tensor(y_pair, dtype=torch.float32)  # (N, 2)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tab[i], self.lidar[i], self.y[i], self.weights[i]


def compute_shared_split(dataset_path, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """Split train/val agrupando por episodio, compartido por los tres γ."""
    data = np.load(dataset_path, allow_pickle=True)
    ep_ids = data["meta_episode_id"].astype(np.int64)
    assert data["observations"].shape[1] == N_OBS

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(ep_ids)), groups=ep_ids))

    train_eps, val_eps = set(ep_ids[train_idx]), set(ep_ids[val_idx])
    assert len(train_eps & val_eps) == 0, "LEAK: episodios compartidos"
    print(f"Split por episodio (val={val_size}, seed={seed}):")
    print(f"  Train: {len(train_idx):>9,} frames ({len(train_eps):>4} episodios)")
    print(f"  Val:   {len(val_idx):>9,} frames ({len(val_eps):>4} episodios)")
    return train_idx, val_idx


def train_for_gamma(gamma, dataset_path, train_idx, val_idx, out_dir, cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    ckpt_dir = Path(out_dir) / f"gamma_{gamma}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    obs_raw = data["observations"].astype(np.float32)
    risk = data[f"gamma_{gamma}"].astype(np.float32)

    obs_tr, risk_tr = obs_raw[train_idx], risk[train_idx]
    obs_va, risk_va = obs_raw[val_idx], risk[val_idx]
    print(f"\n[gamma={gamma}] Train: {len(risk_tr):,} (%>0={100 * (risk_tr > 0).mean():.1f}%) "
          f"| Val: {len(risk_va):,} (%>0={100 * (risk_va > 0).mean():.1f}%)")

    # Etiquetas para las dos cabezas
    y_logp_tr = np.log1p(risk_tr).astype(np.float32)
    y_logp_va = np.log1p(risk_va).astype(np.float32)
    y_bin_tr = (risk_tr > 0).astype(np.float32)
    y_bin_va = (risk_va > 0).astype(np.float32)
    w_tr = (1.0 + ALPHA * risk_tr).astype(np.float32)

    # Escalado de las variables tabulares (se guarda para inferencia)
    scaler = StandardScaler()
    tab_tr = scaler.fit_transform(obs_tr[:, :N_TAB]).astype(np.float32)
    tab_va = scaler.transform(obs_va[:, :N_TAB]).astype(np.float32)
    joblib.dump(scaler, ckpt_dir / "scaler_tabular.pkl")

    ds_tr = RiskDataset(tab_tr, obs_tr[:, N_TAB:],
                        np.stack([y_logp_tr, y_bin_tr], axis=1), w_tr)
    ds_va = RiskDataset(tab_va, obs_va[:, N_TAB:],
                        np.stack([y_logp_va, y_bin_va], axis=1),
                        np.ones_like(risk_va, dtype=np.float32))
    sampler = WeightedRandomSampler(weights=w_tr, num_samples=len(w_tr), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], sampler=sampler,
                       num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"] * 2, shuffle=False,
                       num_workers=4, pin_memory=True)

    model = FullRiskPredictor(RISK_CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] / 20)
    print(f"  Parámetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val, patience_cnt = float("inf"), 0
    history = {"train_total": [], "val_total": [], "val_bce": [], "val_mse_pos": []}
    ckpt_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Entrenamiento ────────────────────────────────────────────────
        model.train()
        tr_total, n_tot = 0.0, 0
        for tab_b, lidar_b, y_b, w_b in dl_tr:
            tab_b, lidar_b = tab_b.to(DEVICE), lidar_b.to(DEVICE)
            y_logp_b, y_bin_b = y_b[:, 0].to(DEVICE), y_b[:, 1].to(DEVICE)
            w_b = w_b.to(DEVICE)
            optimizer.zero_grad()
            logit, mag = model(tab_b, lidar_b)
            loss = (focal_bce(logit, y_bin_b)
                    + cfg["lambda_mag"] * masked_weighted_mse(mag, y_logp_b, y_bin_b, w_b))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(y_bin_b)
            n_tot += n
            tr_total += loss.item() * n
        tr_total /= n_tot

        # ── Validación ───────────────────────────────────────────────────
        model.eval()
        va_bce_sum = va_mse_sum = 0.0
        n_val = n_pos_val = 0
        with torch.no_grad():
            for tab_b, lidar_b, y_b, _ in dl_va:
                tab_b, lidar_b = tab_b.to(DEVICE), lidar_b.to(DEVICE)
                y_logp_b, y_bin_b = y_b[:, 0].to(DEVICE), y_b[:, 1].to(DEVICE)
                logit, mag = model(tab_b, lidar_b)
                va_bce_sum += F.binary_cross_entropy_with_logits(
                    logit, y_bin_b, reduction="sum").item()
                pos_mask = y_bin_b > 0.5
                if pos_mask.any():
                    va_mse_sum += ((mag[pos_mask] - y_logp_b[pos_mask]) ** 2).sum().item()
                    n_pos_val += int(pos_mask.sum().item())
                n_val += len(y_bin_b)
        va_bce = va_bce_sum / n_val
        va_mse_pos = (va_mse_sum / n_pos_val) if n_pos_val > 0 else float("nan")
        va_total = va_bce + cfg["lambda_mag"] * (0.0 if np.isnan(va_mse_pos) else va_mse_pos)
        scheduler.step()

        history["train_total"].append(tr_total)
        history["val_total"].append(va_total)
        history["val_bce"].append(va_bce)
        history["val_mse_pos"].append(va_mse_pos)

        if epoch % 5 == 0 or epoch == 1:
            print(f"   Ep {epoch:3d}/{cfg['epochs']}  loss={tr_total:.5f}  "
                  f"val_total={va_total:.5f} (bce={va_bce:.5f}, mse_pos={va_mse_pos:.5f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if va_total < best_val - 1e-6:
            best_val, patience_cnt = va_total, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"   Early stopping en epoch {epoch} (best={best_val:.5f})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model, scaler, history, best_val, (tab_va, obs_va[:, N_TAB:], risk_va)


@torch.no_grad()
def evaluate_model(model, val_data, batch_size=1024):
    """Métricas de clasificación (y=0 vs y>0) y regresión (sobre y>0)."""
    tab_va, lidar_va, risk_va = val_data
    model.eval()
    preds = []
    for s in range(0, len(tab_va), batch_size):
        e = min(s + batch_size, len(tab_va))
        logit, mag = model(
            torch.tensor(tab_va[s:e], dtype=torch.float32).to(DEVICE),
            torch.tensor(lidar_va[s:e], dtype=torch.float32).to(DEVICE),
        )
        pred = (torch.sigmoid(logit) * torch.expm1(mag)).clamp_min(0.0)
        preds.append(pred.cpu().numpy())
    pred_risk = np.concatenate(preds)

    # ── Clasificación: situaciones seguras (y=0) vs de riesgo (y>0) ───────
    y_bin = (risk_va > 0).astype(np.int32)
    p_bin = (pred_risk > 0).astype(np.int32)
    tp = int(((p_bin == 1) & (y_bin == 1)).sum())
    tn = int(((p_bin == 0) & (y_bin == 0)).sum())
    fp = int(((p_bin == 1) & (y_bin == 0)).sum())
    fn = int(((p_bin == 0) & (y_bin == 1)).sum())
    precision = precision_score(y_bin, p_bin, zero_division=0)
    recall = recall_score(y_bin, p_bin, zero_division=0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ── Regresión restringida a y>0 (evita la distorsión del desbalance) ──
    pos = risk_va > 0
    mae = mean_absolute_error(risk_va[pos], pred_risk[pos])
    mse = mean_squared_error(risk_va[pos], pred_risk[pos])
    r2 = r2_score(risk_va[pos], pred_risk[pos])

    return dict(
        precision=float(precision), recall=float(recall), f1=float(f1),
        TP=tp, TN=tn, FP=fp, FN=fn,
        mae_pos=float(mae), mse_pos=float(mse),
        rmse_pos=float(np.sqrt(mse)), r2_pos=float(r2),
    )


def main():
    args = parse_args()
    cfg = dict(TRAIN_CFG, epochs=args.epochs, batch_size=args.batch_size)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    train_idx, val_idx = compute_shared_split(args.dataset)

    all_metrics = {}
    for gamma in args.gammas:
        print(f"\n{'#' * 70}\n#  gamma = {gamma}\n{'#' * 70}")
        model, _, history, best_val, val_data = train_for_gamma(
            gamma, args.dataset, train_idx, val_idx, out_dir, cfg)
        metrics = evaluate_model(model, val_data)
        metrics["best_val_loss"] = float(best_val)
        metrics["history"] = {k: [float(x) for x in v] for k, v in history.items()}
        all_metrics[str(gamma)] = metrics

        print(f"\n  EVAL gamma={gamma}")
        print(f"    Clasificación: precision={metrics['precision']:.4f} "
              f"recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
        print(f"    TP={metrics['TP']:,} TN={metrics['TN']:,} "
              f"FP={metrics['FP']:,} FN={metrics['FN']:,}")
        print(f"    Regresión (y>0): MAE={metrics['mae_pos']:.4f} "
              f"MSE={metrics['mse_pos']:.4f} RMSE={metrics['rmse_pos']:.4f} "
              f"R²={metrics['r2_pos']:.4f}")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMétricas guardadas en {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
