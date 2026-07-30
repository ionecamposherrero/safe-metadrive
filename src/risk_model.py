"""Modelo supervisado de predicción de riesgo (Fase 2).

Modelo de dos cabezas sobre un tronco común:
  1. Cabeza binaria (focal loss): predice P(risk > 0).
  2. Cabeza de magnitud (Softplus + MSE ponderado): predice log1p(risk),
     entrenada solo sobre muestras positivas.

En inferencia: risk_pred = sigmoid(logit) · expm1(mag).

La observación LidarStateObservation (259 dims) se divide en 19 variables
tabulares (estado del ego + navegación) y 240 lecturas LiDAR, procesadas por
encoders especializados.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TAB = 19
N_LIDAR = 240
N_OBS = N_TAB + N_LIDAR  # 259

# Configuración por defecto de la arquitectura (igual en entrenamiento e inferencia)
RISK_CFG = dict(
    lidar_base_ch=32, lidar_embed_dim=64,
    tab_hidden=[64, 64], tab_dropout=0.30,
    fusion_hidden=[128, 64], fusion_dropout=0.50,
)

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def circular_positional_encoding(n: int) -> torch.Tensor:
    angles = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
    return torch.stack([angles.sin(), angles.cos()], dim=0)


PE = circular_positional_encoding(N_LIDAR)


class CircularConv1d(nn.Module):
    """Convolución 1D con padding circular (preserva la topología del anillo LiDAR)."""

    def __init__(self, in_ch, out_ch, kernel=7, dilation=1):
        super().__init__()
        self.pad = (kernel // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode="circular")
        return F.gelu(self.bn(self.conv(x)))


class LiDAREncoder(nn.Module):
    """Encoder de las 240 lecturas LiDAR: codificación posicional circular +
    3 convoluciones dilatadas {1, 2, 4} con conexiones residuales + pooling
    dual mean+max → embedding de 64 dims."""

    def __init__(self, embed_dim=64, base_ch=32, dropout=0.2):
        super().__init__()
        self.register_buffer("pe", PE)
        self.conv1 = CircularConv1d(3, base_ch, kernel=7, dilation=1)
        self.conv2 = CircularConv1d(base_ch, base_ch * 2, kernel=7, dilation=2)
        self.conv3 = CircularConv1d(base_ch * 2, base_ch * 2, kernel=7, dilation=4)
        self.skip12 = nn.Conv1d(base_ch, base_ch * 2, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(base_ch * 4, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, lidar):
        B = lidar.size(0)
        pe = self.pe.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([lidar.unsqueeze(1), pe], dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + self.skip12(x1)
        x3 = self.conv3(x2) + x2
        x3 = self.drop(x3)
        pooled = torch.cat([x3.mean(-1), x3.max(-1).values], dim=-1)
        return self.proj(pooled)


class TabularEncoder(nn.Module):
    """MLP residual con BatchNorm a la entrada para las 19 variables tabulares."""

    def __init__(self, in_dim=N_TAB, hidden=(64, 64), dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_dim)
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out_dim = prev
        self.res_proj = nn.Linear(in_dim, prev, bias=False) if in_dim != prev else nn.Identity()

    def forward(self, x):
        x_bn = self.input_bn(x)
        return self.mlp(x_bn) + self.res_proj(x_bn)


class FullRiskPredictor(nn.Module):
    """Modelo zero-inflated con dos cabezas sobre un tronco común."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = dict(RISK_CFG) if cfg is None else cfg
        self.lidar_enc = LiDAREncoder(embed_dim=cfg["lidar_embed_dim"],
                                      base_ch=cfg["lidar_base_ch"],
                                      dropout=cfg["tab_dropout"])
        self.tab_enc = TabularEncoder(in_dim=N_TAB, hidden=cfg["tab_hidden"],
                                      dropout=cfg["tab_dropout"])
        fusion_in = cfg["lidar_embed_dim"] + self.tab_enc.out_dim
        # Tronco común
        trunk, prev = [], fusion_in
        for h in cfg["fusion_hidden"]:
            trunk += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                      nn.Dropout(cfg["fusion_dropout"])]
            prev = h
        self.trunk = nn.Sequential(*trunk)
        # Dos cabezas independientes
        self.clf_head = nn.Linear(prev, 1)                                # logit binario
        self.mag_head = nn.Sequential(nn.Linear(prev, 1), nn.Softplus())  # log1p(risk)

    def forward(self, tab, lidar):
        z = self.trunk(torch.cat([self.lidar_enc(lidar), self.tab_enc(tab)], dim=1))
        logit = self.clf_head(z).squeeze(-1)
        mag = self.mag_head(z).squeeze(-1)
        return logit, mag


# ═══════════════════════════════════════════════════════════════════════════
# Funciones de pérdida
# ═══════════════════════════════════════════════════════════════════════════
def focal_bce(logits, targets, alpha=FOCAL_ALPHA, gamma_fl=FOCAL_GAMMA):
    """Focal loss (Lin et al., 2017): reduce el peso de las muestras fáciles."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma_fl * bce).mean()


def masked_weighted_mse(pred, target, mask, weights, eps=1e-8):
    """MSE ponderado calculado SOLO sobre muestras con mask==1."""
    sq = (pred - target) ** 2 * weights * mask
    return sq.sum() / (mask.sum() + eps)


# ═══════════════════════════════════════════════════════════════════════════
# Inferencia: carga del modelo + scaler y predicción sobre observaciones crudas
# ═══════════════════════════════════════════════════════════════════════════
class RiskPredictor:
    """Wrapper de inferencia: carga el checkpoint y el StandardScaler tabular
    y predice el riesgo continuo para observaciones LidarState crudas."""

    def __init__(self, model_path, scaler_path, device="cpu"):
        import joblib
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.model = FullRiskPredictor(RISK_CFG)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)
        self.scaler = joblib.load(scaler_path)

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> float:
        """Riesgo continuo (zero-inflated) para una observación cruda (259,)."""
        tab = self.scaler.transform(obs[:N_TAB].reshape(1, -1)).astype(np.float32)
        lidar = obs[N_TAB:].astype(np.float32).reshape(1, -1)
        tab_t = torch.from_numpy(tab).to(self.device)
        lidar_t = torch.from_numpy(lidar).to(self.device)

        logit, mag = self.model(tab_t, lidar_t)
        p_pos = torch.sigmoid(logit)
        pred_risk = (p_pos * torch.expm1(mag)).clamp_min(0.0)
        return float(pred_risk.item())

    @torch.no_grad()
    def predict_batch(self, obs: np.ndarray) -> np.ndarray:
        """Riesgo continuo para un lote de observaciones (N, 259)."""
        tab = self.scaler.transform(obs[:, :N_TAB]).astype(np.float32)
        lidar = obs[:, N_TAB:].astype(np.float32)
        logit, mag = self.model(
            torch.from_numpy(tab).to(self.device),
            torch.from_numpy(lidar).to(self.device),
        )
        p_pos = torch.sigmoid(logit)
        pred = (p_pos * torch.expm1(mag)).clamp_min(0.0)
        return pred.cpu().numpy()
