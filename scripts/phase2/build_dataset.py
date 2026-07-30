"""Fase 2 (paso 2) — Etiquetado y fusión: dataset multi-label.

Carga los cuatro datasets base (PPO, SAC, PPOLag, SACLag), los unifica y
aplica los esquemas de etiquetado sobre exactamente los mismos episodios:

  - Gamma (usado): riesgo por descuento temporal hacia atrás
        risk_γ(t) = collision(t+1) + γ · risk_γ(t+1),  γ ∈ {0.8, 0.9, 0.95}
  - K-binario: 1 si hay colisión en los próximos K pasos, K ∈ {5, 10, 20}
  - K-progresivo: suma de colisiones futuras con peso lineal decreciente

Resultado: dataset_final_multilabel.npz con todas las etiquetas por frame
y metadatos de política/modelo/nivel.

Uso:
    python scripts/phase2/build_dataset.py
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

GAMMA_LIST = [0.8, 0.9, 0.95]
K_LIST = [5, 10, 20]
POLICIES = ["PPO", "SAC", "PPOLag", "SACLag"]

LABEL_KEYS = (
    [f"gamma_{g}" for g in GAMMA_LIST] +
    [f"k_binario_{k}" for k in K_LIST] +
    [f"k_progresivo_{k}" for k in K_LIST]
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default=str(ROOT / "results" / "phase2" / "dataset"),
                   help="Carpeta con los dataset_base_<POLICY>.pkl")
    p.add_argument("--output", default=None,
                   help="Ruta del .npz final (por defecto, dataset-dir/dataset_final_multilabel.npz)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Funciones de etiquetado
# ══════════════════════════════════════════════════════════════════════════
def label_gamma(collisions: np.ndarray, gamma: float) -> np.ndarray:
    """Retorno de riesgo descontado hacia atrás: risk(t) = c(t+1) + γ·risk(t+1).

    El último paso se fija a 0 (sin información futura).
    """
    T = len(collisions)
    risk = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in range(T - 2, -1, -1):
        running = collisions[t + 1] + gamma * running
        risk[t] = running
    risk[T - 1] = 0.0
    return risk


def label_k_binario(collisions: np.ndarray, K: int) -> np.ndarray:
    """1 si ocurre alguna colisión en los próximos K pasos, 0 en caso contrario."""
    T = len(collisions)
    label = np.zeros(T, dtype=np.float32)
    for t in range(T - 1):
        window = collisions[t + 1: t + K + 1]
        label[t] = 1.0 if window.max() > 0 else 0.0
    return label


def label_k_progresivo(collisions: np.ndarray, K: int) -> np.ndarray:
    """Suma ponderada con decaimiento lineal: colisiones próximas pesan más.

    label(t) = Σ_{j=1}^{K} collision(t+j) · (K − j + 1) / K
    """
    T = len(collisions)
    weights = np.array([(K - j) / K for j in range(K)], dtype=np.float32)
    label = np.zeros(T, dtype=np.float32)
    for t in range(T - 1):
        window = collisions[t + 1: t + K + 1].astype(np.float32)
        w = weights[:len(window)]
        label[t] = float(np.dot(window, w))
    return label


def compute_all_labels(collisions: np.ndarray) -> dict:
    labels = {}
    for g in GAMMA_LIST:
        labels[f"gamma_{g}"] = label_gamma(collisions, g)
    for k in K_LIST:
        labels[f"k_binario_{k}"] = label_k_binario(collisions, k)
    for k in K_LIST:
        labels[f"k_progresivo_{k}"] = label_k_progresivo(collisions, k)
    return labels


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_file = Path(args.output) if args.output else dataset_dir / "dataset_final_multilabel.npz"

    # ── 1 · Carga de datasets base ────────────────────────────────────────
    all_episodes = []
    for policy in POLICIES:
        fpath = dataset_dir / f"dataset_base_{policy}.pkl"
        if not fpath.exists():
            raise FileNotFoundError(
                f"No existe {fpath}. Genera primero los datasets base con collect_dataset.py")
        with open(fpath, "rb") as f:
            episodes = pickle.load(f)
        for ep in episodes:
            assert ep["policy"] == policy, f"Inconsistencia: {ep['policy']} ≠ {policy}"
        all_episodes.extend(episodes)
        n_frames = sum(len(e["observations"]) for e in episodes)
        print(f"  {policy:<8}: {len(episodes):>6,} episodios | {n_frames:>10,} frames")

    print(f"\nTotal episodios fusionados: {len(all_episodes):,}")
    print("Distribución por política:", dict(Counter(e["policy"] for e in all_episodes)))

    # ── 2 · Etiquetado frame a frame ──────────────────────────────────────
    all_obs, all_cols = [], []
    all_labels = {k: [] for k in LABEL_KEYS}
    meta = {k: [] for k in ("policy", "model", "seed", "steps", "level",
                            "density", "accident", "episode_idx", "episode_id")}
    ep_id_counter = 0

    print(f"\nProcesando {len(all_episodes):,} episodios...")
    for ep in all_episodes:
        obs = ep["observations"]
        cols = ep["collisions"]
        T = len(obs)
        if T == 0:
            continue

        ep_labels = compute_all_labels(cols)
        all_obs.append(obs)
        all_cols.append(cols)
        for k in LABEL_KEYS:
            all_labels[k].append(ep_labels[k])

        meta["policy"].extend([ep["policy"]] * T)
        meta["model"].extend([ep["model"]] * T)
        meta["seed"].extend([ep["seed"]] * T)
        meta["steps"].extend([ep["steps"]] * T)
        meta["level"].extend([ep["level"]] * T)
        meta["density"].extend([ep["traffic_density"]] * T)
        meta["accident"].extend([ep["accident_prob"]] * T)
        meta["episode_idx"].extend([ep["episode_idx"]] * T)
        meta["episode_id"].extend([ep_id_counter] * T)
        ep_id_counter += 1

    all_obs_np = np.concatenate(all_obs, axis=0)
    all_cols_np = np.concatenate(all_cols, axis=0)
    label_arrays = {k: np.concatenate(all_labels[k], axis=0) for k in LABEL_KEYS}

    N = len(all_obs_np)
    print(f"\nDataset multi-label construido:")
    print(f"  Frames totales:  {N:>10,}")
    print(f"  Episodios:       {ep_id_counter:>10,}")
    print(f"  Dimensión obs:   {all_obs_np.shape[1]:>10}")

    # ── 3 · Guardado ──────────────────────────────────────────────────────
    save_kwargs = {
        "observations": all_obs_np,
        "collisions": all_cols_np,
        "meta_policy": np.array(meta["policy"]),
        "meta_model": np.array(meta["model"]),
        "meta_seed": np.array(meta["seed"], dtype=np.int32),
        "meta_steps": np.array(meta["steps"], dtype=np.int64),
        "meta_level": np.array(meta["level"]),
        "meta_density": np.array(meta["density"], dtype=np.float32),
        "meta_accident": np.array(meta["accident"], dtype=np.float32),
        "meta_episode_idx": np.array(meta["episode_idx"], dtype=np.int32),
        "meta_episode_id": np.array(meta["episode_id"], dtype=np.int32),
    }
    save_kwargs.update(label_arrays)
    np.savez_compressed(output_file, **save_kwargs)

    size_mb = os.path.getsize(output_file) / 1024 ** 2
    print(f"\nDataset guardado: {output_file} ({size_mb:.1f} MB)")

    # ── 4 · Verificación de coherencia ────────────────────────────────────
    g8, g9, g95 = label_arrays["gamma_0.8"], label_arrays["gamma_0.9"], label_arrays["gamma_0.95"]
    assert np.all(g8 >= g9 - 1e-6), "ERROR: gamma_0.8 debería ser >= gamma_0.9"
    assert np.all(g9 >= g95 - 1e-6), "ERROR: gamma_0.9 debería ser >= gamma_0.95"
    print("Orden gamma (0.8 ≥ 0.9 ≥ 0.95): OK")

    for pol in POLICIES:
        mask = save_kwargs["meta_policy"] == pol
        print(f"  {pol:<8}: {len(np.unique(save_kwargs['meta_episode_id'][mask])):>6,} episodios | "
              f"% frames con riesgo>0 (γ=0.9): {100 * (g9[mask] > 0).mean():.1f}%")


if __name__ == "__main__":
    main()
