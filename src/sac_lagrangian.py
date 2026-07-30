"""SAC Lagrangiano (SAC-Lag) sobre Stable-Baselines3.
"""

from typing import NamedTuple, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update


# ═══════════════════════════════════════════════════════════════════════════
# 1) Replay buffer que también almacena el coste extraído de info["cost"]
# ═══════════════════════════════════════════════════════════════════════════
class CostReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor


class CostReplayBuffer(ReplayBuffer):
    """ReplayBuffer que añade un canal de coste por transición."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        # Leer coste ANTES de super().add (porque super avanza self.pos)
        costs = np.array(
            [float(info.get("cost", 0.0)) for info in infos],
            dtype=np.float32,
        )
        self.costs[self.pos] = costs
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> CostReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.costs[batch_inds, env_indices].reshape(-1, 1),
        )
        return CostReplayBufferSamples(*tuple(map(self.to_torch, data)))


# ═══════════════════════════════════════════════════════════════════════════
# 2) Crítico de coste Q_c — MLP gemelo independiente (misma arq. que Q_r)
# ═══════════════════════════════════════════════════════════════════════════
class CostCritic(nn.Module):
    """Ensemble de redes Q para estimar el coste descontado."""

    def __init__(self, obs_dim, act_dim, net_arch, n_critics=2, activation_fn=nn.ReLU):
        super().__init__()
        self.n_critics = n_critics
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            layers = []
            prev = obs_dim + act_dim
            for h in net_arch:
                layers.append(nn.Linear(prev, h))
                layers.append(activation_fn())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.q_networks.append(nn.Sequential(*layers))

    def forward(self, obs, action):
        x = th.cat([obs, action], dim=1)
        return tuple(q(x) for q in self.q_networks)

    def set_training_mode(self, mode: bool):
        self.train(mode)


# ═══════════════════════════════════════════════════════════════════════════
# 3) Clase principal SAC-Lag
# ═══════════════════════════════════════════════════════════════════════════
class SACLagrangian(SAC):
    """SAC con restricción vía multiplicador de Lagrange (SAC-Lag).

    Hereda de SAC y añade:
      - cost_critic y cost_critic_target
      - log_lambda (Parameter) y lambda_optimizer
      - CostReplayBuffer como buffer por defecto
    """

    def __init__(
        self,
        *args,
        cost_limit: float = 1.0,
        cost_gamma: float = 0.99,
        lambda_lr: float = 3e-5,
        init_lambda: float = 0.01,
        lambda_max: float = 100.0,
        n_cost_critics: int = 2,
        cost_agg: str = "mean",
        cost_net_arch: Optional[list] = None,
        normalize_actor_loss: bool = True,
        **kwargs,
    ):
        # Guardar hiperparámetros antes de super().__init__ porque
        # éste puede invocar _setup_model() → _setup_cost_components()
        self.cost_limit = float(cost_limit)
        self.cost_gamma = float(cost_gamma)
        self.lambda_lr = float(lambda_lr)
        self.init_lambda = float(init_lambda)
        self.lambda_max = float(lambda_max)
        self.n_cost_critics = int(n_cost_critics)
        self.cost_agg = str(cost_agg)
        self.cost_net_arch = list(cost_net_arch) if cost_net_arch is not None else [256, 256]
        self.normalize_actor_loss = bool(normalize_actor_loss)

        # Forzar CostReplayBuffer por defecto
        if "replay_buffer_class" not in kwargs or kwargs["replay_buffer_class"] is None:
            kwargs["replay_buffer_class"] = CostReplayBuffer

        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._setup_cost_components()

    def _setup_cost_components(self) -> None:
        """Crea crítico de coste, su target, optimizador y el multiplicador λ."""
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))

        # Crítico de coste (online) + target
        self.cost_critic = CostCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            net_arch=self.cost_net_arch,
            n_critics=self.n_cost_critics,
        ).to(self.device)

        self.cost_critic_target = CostCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            net_arch=self.cost_net_arch,
            n_critics=self.n_cost_critics,
        ).to(self.device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_target.set_training_mode(False)
        for p in self.cost_critic_target.parameters():
            p.requires_grad = False

        # Optimizador del crítico de coste — misma LR base que el actor/crítico
        base_lr = (
            self.lr_schedule(1.0)
            if callable(self.lr_schedule)
            else float(self.lr_schedule)
        )
        self.cost_critic_optimizer = th.optim.Adam(
            self.cost_critic.parameters(), lr=base_lr
        )

        # Multiplicador de Lagrange en log-espacio (λ = exp(log_λ))
        init_log_lam = float(np.log(max(self.init_lambda, 1e-8)))
        self.log_lambda = nn.Parameter(
            th.tensor(init_log_lam, dtype=th.float32, device=self.device),
            requires_grad=True,
        )
        self.lambda_optimizer = th.optim.Adam([self.log_lambda], lr=self.lambda_lr)

    # ────────────────────────────────────────────────────────────────────
    # Helpers de agregación para el crítico gemelo de coste
    # ────────────────────────────────────────────────────────────────────
    def _aggregate_cost(self, q_c_tuple):
        """Combina los Q_c del ensemble según self.cost_agg → (B, 1)."""
        q_c_stack = th.cat(q_c_tuple, dim=1)  # (B, n_cost_critics)
        if self.cost_agg == "max":
            val, _ = th.max(q_c_stack, dim=1, keepdim=True)
        elif self.cost_agg == "min":
            val, _ = th.min(q_c_stack, dim=1, keepdim=True)
        else:  # "mean"
            val = th.mean(q_c_stack, dim=1, keepdim=True)
        return val

    # ────────────────────────────────────────────────────────────────────
    # Paso de entrenamiento (sobreescribe SAC.train)
    # ────────────────────────────────────────────────────────────────────
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self.cost_critic.set_training_mode(True)

        # Actualizar LR de los optimizadores (λ mantiene su LR dedicada)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.cost_critic_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, cost_critic_losses = [], [], []
        lambda_losses, lambda_values, q_c_means = [], [], []

        for _ in range(gradient_steps):
            # ── Muestrear del replay buffer (con costes) ─────────────────
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Acciones de la política actual (para entropía y actor loss)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # ── α (entropy coef) update ─────────────────────────────────
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ── Crítico de recompensa (SAC estándar) ─────────────────────
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q_values) for q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ── Crítico de coste Q_c ────────────────────────────────────
            with th.no_grad():
                # next_actions reutilizadas del bloque anterior
                next_q_c = self._aggregate_cost(
                    self.cost_critic_target(replay_data.next_observations, next_actions)
                )
                # NOTA: el crítico de coste NO usa entropía en su TD target
                target_q_c_values = replay_data.costs + (1 - replay_data.dones) * self.cost_gamma * next_q_c

            current_q_c_values = self.cost_critic(replay_data.observations, replay_data.actions)
            cost_critic_loss = 0.5 * sum(F.mse_loss(q_c, target_q_c_values) for q_c in current_q_c_values)
            cost_critic_losses.append(cost_critic_loss.item())

            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic_optimizer.step()

            # ── Actor update (con término Lagrangiano) ───────────────────
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            q_c_pi = self._aggregate_cost(self.cost_critic(replay_data.observations, actions_pi))

            lam = th.exp(self.log_lambda.detach()).clamp(max=self.lambda_max)  # scalar

            actor_loss = (ent_coef * log_prob - min_qf_pi + lam * q_c_pi).mean()
            if self.normalize_actor_loss:
                actor_loss = actor_loss / (1.0 + lam.item())

            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ── Multiplicador de Lagrange λ (ascenso) ────────────────────
            with th.no_grad():
                j_c_estimate = q_c_pi.mean()          # ~ E[Q_c(s, π(s))]
            q_c_means.append(j_c_estimate.item())

            # Pérdida: minimizamos −log_λ · (J_c − d) ⇒ log_λ ← log_λ + lr·(J_c − d)
            constraint_violation = j_c_estimate - self.cost_limit
            lambda_loss = -self.log_lambda * constraint_violation.detach()
            lambda_losses.append(lambda_loss.item())

            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()

            # Cota superior (evita explosión) y cota inferior suave
            with th.no_grad():
                self.log_lambda.clamp_(max=float(np.log(self.lambda_max)), min=-20.0)

            lambda_values.append(float(th.exp(self.log_lambda.detach()).item()))

            # ── Soft update de los targets (reward + cost) ───────────────
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self._n_updates += 1

        # ── Logging ─────────────────────────────────────────────────────
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/cost_critic_loss", float(np.mean(cost_critic_losses)))
        self.logger.record("train/lambda", float(np.mean(lambda_values)))
        self.logger.record("train/lambda_loss", float(np.mean(lambda_losses)))
        self.logger.record("train/q_c_mean", float(np.mean(q_c_means)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))

    # ────────────────────────────────────────────────────────────────────
    # save / load — exponer módulos y tensores nuevos
    # ────────────────────────────────────────────────────────────────────
    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "cost_critic",
            "cost_critic_target",
            "log_lambda",
        ]

    def _get_torch_save_params(self):
        state_dicts, pytorch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts) + [
            "cost_critic",
            "cost_critic_target",
            "cost_critic_optimizer",
            "lambda_optimizer",
        ]
        pytorch_vars = list(pytorch_vars or []) + ["log_lambda"]
        return state_dicts, pytorch_vars
