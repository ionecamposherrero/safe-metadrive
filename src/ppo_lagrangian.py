"""PPO Lagrangiano (PPO-Lag) sobre Stable-Baselines3.
"""

import math
from typing import NamedTuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor


# ═══════════════════════════════════════════════════════════════════════════
# Buffer de rollout que además almacena el coste y V_c por paso
# ═══════════════════════════════════════════════════════════════════════════
class CostRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    cost_advantages: torch.Tensor
    cost_returns: torch.Tensor
    old_cost_values: torch.Tensor


class CostRolloutBuffer(RolloutBuffer):
    """RolloutBuffer extendido: por cada paso guarda además el coste (desde
    info["cost"]) y el valor V_c(s). Calcula GAE para el coste con
    (cost_gamma, cost_gae_lambda) independientes de los de la recompensa.
    """

    def __init__(self, *args, cost_gamma=0.99, cost_gae_lambda=0.95, **kwargs):
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, cost=None, cost_value=None, **kwargs):
        # Capturar la posición ANTES de que super().add la incremente
        pos = self.pos
        super().add(*args, **kwargs)
        if cost is not None:
            self.costs[pos] = np.asarray(cost, dtype=np.float32).copy()
        if cost_value is not None:
            if isinstance(cost_value, torch.Tensor):
                cost_value = cost_value.detach().cpu().numpy()
            self.cost_values[pos] = np.asarray(cost_value, dtype=np.float32).flatten().copy()

    def compute_cost_returns_and_advantage(self, last_cost_values, dones):
        """GAE para el coste — análogo a compute_returns_and_advantage de SB3."""
        if isinstance(last_cost_values, torch.Tensor):
            last_cost_values = last_cost_values.detach().cpu().numpy().flatten()
        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
                next_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.cost_values[step + 1]
            delta = (
                self.costs[step]
                + self.cost_gamma * next_values * next_non_terminal
                - self.cost_values[step]
            )
            last_gae_lam = (
                delta
                + self.cost_gamma * self.cost_gae_lambda * next_non_terminal * last_gae_lam
            )
            self.cost_advantages[step] = last_gae_lam
        self.cost_returns = self.cost_advantages + self.cost_values

    def get(self, batch_size=None):
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            _tensor_names = [
                "observations", "actions", "values", "log_probs",
                "advantages", "returns",
                "cost_values", "cost_advantages", "cost_returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_cost_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_cost_samples(self, batch_inds):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
        )
        return CostRolloutBufferSamples(*tuple(map(self.to_torch, data)))


# ═══════════════════════════════════════════════════════════════════════════
# Red de valor del coste (V_c) — MLP estándar
# ═══════════════════════════════════════════════════════════════════════════
class CostValueNet(nn.Module):
    def __init__(self, obs_dim: int, net_arch=(256, 256), activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in net_arch:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# PPOLagrangian: extiende PPO con crítico de coste V_c y multiplicador λ
# ═══════════════════════════════════════════════════════════════════════════
class PPOLagrangian(PPO):
    """PPO con restricción de coste mediante multiplicador de Lagrange.

    Detalles de la implementación:
      [1] lambda_optimizer = SGD (no Adam) → λ responde a la magnitud de la
          violación, no se mueve a velocidad fija.
      [2] J_c se estima como media de coste UNDISCOUNTED por episodio terminado
          en el rollout, de forma que cost_limit=1 significa "máximo 1 colisión
          por episodio en promedio".
      [3] A_r y A_c se normalizan a media 0, std 1 POR SEPARADO antes de
          combinar A = A_r − λ·A_c.
      [4] cost_critic_optimizer NO recibe el schedule de LR del policy →
          cost_vf_lr permanece fijo.
      [5] norm_actor_loss=True divide combined_adv por (1+λ) sin re-normalizar
          después (Stooke et al.).
    """

    def __init__(
        self,
        *args,
        cost_limit: float = 1.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        lambda_lr: float = 5e-2,
        init_lambda: float = 0.5,
        lambda_max: float = 100.0,
        cost_vf_lr: float = 3e-4,
        cost_net_arch: Optional[List[int]] = None,
        norm_actor_loss: bool = True,
        **kwargs,
    ):
        self.cost_limit = cost_limit
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.lambda_lr = lambda_lr
        self.init_lambda = init_lambda
        self.lambda_max = lambda_max
        self.cost_vf_lr = cost_vf_lr
        self.cost_net_arch = list(cost_net_arch) if cost_net_arch is not None else [256, 256]
        self.norm_actor_loss = norm_actor_loss
        # Buffer para costes episódicos del rollout actual
        self._rollout_episode_costs: List[float] = []
        self._running_ep_costs: Optional[np.ndarray] = None
        super().__init__(*args, **kwargs)

    # ── Setup ─────────────────────────────────────────────────────────────
    def _setup_model(self) -> None:
        super()._setup_model()
        # Reemplazar el rollout buffer por uno con coste
        self.rollout_buffer = CostRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
        )
        self._setup_cost_components()

    def _setup_cost_components(self) -> None:
        obs_dim = int(np.prod(self.observation_space.shape))
        self.cost_critic = CostValueNet(obs_dim, net_arch=self.cost_net_arch).to(self.device)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=self.cost_vf_lr
        )
        # λ en log-space
        init_log = math.log(max(self.init_lambda, 1e-8))
        self.log_lambda = nn.Parameter(
            torch.tensor(init_log, dtype=torch.float32, device=self.device)
        )
        # SGD: el paso en log_λ es proporcional a la violación (J_c − d)
        self.lambda_optimizer = torch.optim.SGD([self.log_lambda], lr=self.lambda_lr)

        # Tracker de coste episódico durante rollouts
        self._running_ep_costs = np.zeros(self.n_envs, dtype=np.float32)

    # ── Recolección de rollouts con coste ─────────────────────────────────
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        self.cost_critic.eval()

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        # Mantenemos _running_ep_costs entre rollouts (un episodio puede cruzar
        # la frontera de un rollout) pero limpiamos la lista de terminados.
        self._rollout_episode_costs = []
        if self._running_ep_costs is None or len(self._running_ep_costs) != env.num_envs:
            self._running_ep_costs = np.zeros(env.num_envs, dtype=np.float32)

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                cost_values = self.cost_critic(obs_tensor)
            actions_np = actions.cpu().numpy()

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions_np, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Extraer coste de infos
            costs = np.array([info.get("cost", 0.0) for info in infos], dtype=np.float32)

            # Tracking de coste episódico RAW (sin bootstrap): J_c = E[Σ c_t]
            self._running_ep_costs += costs

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            acts_to_store = actions_np
            if isinstance(self.action_space, spaces.Discrete):
                acts_to_store = actions_np.reshape(-1, 1)

            # Bootstrap en timeouts (afecta a costs[idx] solo para GAE de V_c)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_cost_value = self.cost_critic(terminal_obs)
                    rewards[idx] += self.gamma * terminal_value.item()
                    costs[idx] += self.cost_gamma * terminal_cost_value.item()

            # Detectar episodios completados y guardar su coste raw
            for idx, d in enumerate(dones):
                if d:
                    self._rollout_episode_costs.append(float(self._running_ep_costs[idx]))
                    self._running_ep_costs[idx] = 0.0

            rollout_buffer.add(
                self._last_obs,
                acts_to_store,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                cost=costs,
                cost_value=cost_values,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Bootstrap final
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            cost_values = self.cost_critic(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_cost_returns_and_advantage(
            last_cost_values=cost_values, dones=dones
        )

        callback.on_rollout_end()
        return True

    # ── Paso de entrenamiento ─────────────────────────────────────────────
    def train(self) -> None:
        self.policy.set_training_mode(True)
        self.cost_critic.train()
        # Solo el policy_optimizer recibe el schedule de LR; cost_vf_lr es fijo
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Valor actual de λ (fijo durante todas las épocas de este train())
        lam = self.log_lambda.detach().exp().item()

        entropy_losses, pg_losses, value_losses, cost_value_losses = [], [], [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Ventaja combinada: A_r y A_c normalizadas por separado,
                # A = A_r − λ·A_c, y división opcional por (1+λ).
                rew_adv = rollout_data.advantages
                cost_adv = rollout_data.cost_advantages

                if self.normalize_advantage:
                    if len(rew_adv) > 1:
                        rew_adv = (rew_adv - rew_adv.mean()) / (rew_adv.std() + 1e-8)
                    if len(cost_adv) > 1:
                        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)

                combined_adv = rew_adv - lam * cost_adv
                if self.norm_actor_loss:
                    combined_adv = combined_adv / (1.0 + lam)

                # ── Pérdida de política PPO clippeada ────────────────────
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = combined_adv * ratio
                policy_loss_2 = combined_adv * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # ── Pérdida de V_r (valor de la recompensa) ──────────────
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # ── Pérdida de entropía ──────────────────────────────────
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ── Pérdida total del policy ─────────────────────────────
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Early stopping por KL
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                        .cpu()
                        .numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    break

                # ── Paso de optimización del policy ──────────────────────
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                # ── Paso de optimización del crítico de coste ────────────
                # (forward fresh para no reutilizar grafo liberado)
                cost_values_pred = self.cost_critic(rollout_data.observations).flatten()
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred)
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.cost_critic.parameters(), self.max_grad_norm
                )
                self.cost_critic_optimizer.step()
                cost_value_losses.append(cost_value_loss.item())

            self._n_updates += 1
            if not continue_training:
                break

        # ── Actualización del multiplicador de Lagrange ──────────────────
        # J_c = coste medio por episodio terminado en el rollout
        if len(self._rollout_episode_costs) > 0:
            jc_estimate = float(np.mean(self._rollout_episode_costs))
            n_episodes_used = len(self._rollout_episode_costs)
        else:
            # Fallback: si ningún episodio terminó en el rollout, usar la
            # media del buffer (caso raro: rollouts cortos vs episodios largos)
            jc_estimate = float(self.rollout_buffer.cost_returns.mean())
            n_episodes_used = 0

        jc_tensor = torch.tensor(jc_estimate, dtype=torch.float32, device=self.device)
        violation = (jc_tensor - self.cost_limit).detach()
        # loss = -log_lambda * violation (ascenso → incrementa λ si violation>0)
        lambda_loss = -(self.log_lambda * violation)
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        # Salvaguarda: limitar el paso máximo en log_λ por iteración
        torch.nn.utils.clip_grad_norm_([self.log_lambda], max_norm=50.0)
        self.lambda_optimizer.step()

        with torch.no_grad():
            self.log_lambda.clamp_(min=-20.0, max=math.log(self.lambda_max))

        # ── Logging ──────────────────────────────────────────────────────
        explained_var = 1 - np.var(
            self.rollout_buffer.returns.flatten() - self.rollout_buffer.values.flatten()
        ) / (np.var(self.rollout_buffer.returns.flatten()) + 1e-8)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/lambda", float(self.log_lambda.detach().exp().item()))
        self.logger.record("train/J_c", jc_estimate)
        self.logger.record("train/J_c_n_episodes", n_episodes_used)
        self.logger.record("train/J_c_violation", float(violation.item()))
        self.logger.record("train/cost_limit", self.cost_limit)

    # ── Save / load ──────────────────────────────────────────────────────
    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "cost_critic", "log_lambda",
        ]

    def _get_torch_save_params(self):
        state_dicts, pytorch_vars = super()._get_torch_save_params()
        state_dicts = list(state_dicts) + [
            "cost_critic",
            "cost_critic_optimizer",
            "lambda_optimizer",
        ]
        pytorch_vars = list(pytorch_vars) if pytorch_vars is not None else []
        pytorch_vars = pytorch_vars + ["log_lambda"]
        return state_dicts, pytorch_vars
