"""Hiperparámetros y construcción de los modelos PPO, SAC, PPO-Lag y SAC-Lag.
"""

from stable_baselines3 import PPO, SAC

from src.ppo_lagrangian import PPOLagrangian
from src.sac_lagrangian import SACLagrangian

# ══════════════════════════════════════════════════════════════════════════
# PPO
# ══════════════════════════════════════════════════════════════════════════
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_N_EPOCHS = 8
PPO_MINI_BATCH = 1024
PPO_CLIP_EPS = 0.2
PPO_TARGET_KL = 0.025
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.25
PPO_MAX_GRAD_NORM = 0.5
PPO_N_STEPS = 4096


def ppo_lr_schedule(progress_remaining: float) -> float:
    """LR con decay lineal: 3e-4 → 5e-5."""
    return 5e-5 + (3e-4 - 5e-5) * progress_remaining


# ══════════════════════════════════════════════════════════════════════════
# SAC
# ══════════════════════════════════════════════════════════════════════════
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_LEARNING_RATE = 3e-4
SAC_LEARNING_STARTS = 10_000
SAC_BUFFER_SIZE = 1_000_000
SAC_BATCH_SIZE = 512
SAC_INIT_ALPHA = 0.1
SAC_TARGET_ENTROPY = -2.0
SAC_GRADIENT_STEPS = 1


def sac_lr_schedule(progress_remaining: float) -> float:
    """LR con decay lineal hasta el 5% del valor inicial."""
    return SAC_LEARNING_RATE * max(progress_remaining, 0.05)


# ══════════════════════════════════════════════════════════════════════════
# Hiperparámetros Lagrangianos
# ══════════════════════════════════════════════════════════════════════════
PPOLAG_KWARGS = dict(
    cost_limit=1.0,
    cost_gamma=0.99,
    cost_gae_lambda=0.92,
    lambda_lr=1e-3,
    init_lambda=0.5,
    lambda_max=100.0,
    cost_vf_lr=3e-4,
    cost_net_arch=[256, 256],
    norm_actor_loss=True,
)

SACLAG_KWARGS = dict(
    cost_limit=1.0,
    cost_gamma=0.99,
    lambda_lr=3e-5,
    init_lambda=0.01,
    lambda_max=100.0,
    n_cost_critics=2,
    cost_agg="mean",
    cost_net_arch=[256, 256],
    normalize_actor_loss=True,
)

# ══════════════════════════════════════════════════════════════════════════
# Registro de algoritmos
# ══════════════════════════════════════════════════════════════════════════
ALGOS = ["ppo", "sac", "ppo_lag", "sac_lag"]

# Clase SB3 usada para cargar/guardar cada algoritmo
ALGO_CLASS = {
    "ppo": PPO,
    "sac": SAC,
    "ppo_lag": PPOLagrangian,
    "sac_lag": SACLagrangian,
}

# PPO (on-policy) normaliza también la recompensa; SAC solo la observación
ALGO_NORM_REWARD = {"ppo": True, "ppo_lag": True, "sac": False, "sac_lag": False}

# Dispositivo recomendado para cada algoritmo (MLP pequeño: PPO rinde más en CPU)
ALGO_DEVICE = {"ppo": "cpu", "ppo_lag": "cpu", "sac": "auto", "sac_lag": "auto"}

N_ENVS = 8
HORIZON = 1000
LOG_INTERVAL = 8000


def make_model(algo: str, env, seed: int, device: str = None, verbose: int = 1):
    """Construye el modelo SB3 con los hiperparámetros optimizados del TFG."""
    algo = algo.lower()
    if device is None:
        device = ALGO_DEVICE[algo]

    if algo in ("ppo", "ppo_lag"):
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        common = dict(
            policy_kwargs=policy_kwargs,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            learning_rate=ppo_lr_schedule,
            clip_range=PPO_CLIP_EPS,
            vf_coef=PPO_VF_COEF,
            ent_coef=PPO_ENT_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            target_kl=PPO_TARGET_KL,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_MINI_BATCH,
            n_epochs=PPO_N_EPOCHS,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        if algo == "ppo":
            return PPO("MlpPolicy", env, **common)
        return PPOLagrangian("MlpPolicy", env, **common, **PPOLAG_KWARGS)

    if algo in ("sac", "sac_lag"):
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        common = dict(
            policy_kwargs=policy_kwargs,
            gradient_steps=SAC_GRADIENT_STEPS,
            gamma=SAC_GAMMA,
            tau=SAC_TAU,
            learning_rate=sac_lr_schedule,
            learning_starts=SAC_LEARNING_STARTS,
            buffer_size=SAC_BUFFER_SIZE,
            batch_size=SAC_BATCH_SIZE,
            ent_coef=f"auto_{SAC_INIT_ALPHA}",
            target_entropy=SAC_TARGET_ENTROPY,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        if algo == "sac":
            return SAC("MlpPolicy", env, **common)
        return SACLagrangian("MlpPolicy", env, **common, **SACLAG_KWARGS)

    raise ValueError(f"Algoritmo desconocido: {algo!r}. Opciones: {ALGOS}")


def make_default_model(algo: str, env, seed: int, verbose: int = 1):
    """Modelo con los hiperparámetros POR DEFECTO de SB3 (Fase 0, config base)."""
    algo = algo.lower()
    if algo == "ppo":
        return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=verbose)
    if algo == "sac":
        return SAC("MlpPolicy", env, seed=seed, device="auto", verbose=verbose)
    raise ValueError(f"La Fase 0 solo admite 'ppo' o 'sac', no {algo!r}")
