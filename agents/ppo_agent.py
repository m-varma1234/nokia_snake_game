"""
PPO Agent — Proximal Policy Optimization (clip variant, discrete actions).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, x: torch.Tensor):
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, dist.entropy()


class PPOAgent:
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 512,
        device: str | None = None,
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.rollout_steps = rollout_steps

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else "cpu"
            )
        )

        self.net       = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Rollout buffers
        self._clear_rollout()

    def _clear_rollout(self):
        self._states     = []
        self._actions    = []
        self._log_probs  = []
        self._rewards    = []
        self._values     = []
        self._dones      = []

    # ------------------------------------------------------------------ act
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value, _ = self.net.get_action(s)
        if training:
            self._states.append(state)
            self._actions.append(action)
            self._log_probs.append(log_prob.item())
            self._values.append(value.item())
        return action

    def store_reward(self, reward: float, done: bool):
        self._rewards.append(reward)
        self._dones.append(float(done))

    # ----------------------------------------------------------- GAE + learn
    def learn(self, last_value: float = 0.0) -> float | None:
        if len(self._rewards) < self.batch_size:
            return None

        # Compute GAE advantages
        rewards  = self._rewards
        values   = self._values + [last_value]
        dones    = self._dones
        adv      = []
        gae      = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv.insert(0, gae)

        adv_t      = torch.tensor(adv,             dtype=torch.float32, device=self.device)
        returns_t  = adv_t + torch.tensor(values[:-1], dtype=torch.float32, device=self.device)
        states_t   = torch.tensor(np.array(self._states),  dtype=torch.float32, device=self.device)
        actions_t  = torch.tensor(self._actions,  dtype=torch.long,    device=self.device)
        old_lp_t   = torch.tensor(self._log_probs, dtype=torch.float32, device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        n = len(self._rewards)
        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                logits, values_pred = self.net(states_t[mb])
                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(actions_t[mb])
                entropy  = dist.entropy().mean()

                ratio    = (new_lp - old_lp_t[mb]).exp()
                surr1    = ratio * adv_t[mb]
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_pred, returns_t[mb])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        steps = self.n_epochs * max(1, n // self.batch_size)
        self._clear_rollout()
        return total_loss / steps

    # ----------------------------------------------------------- save/load
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, path)
        print(f"Model saved → {path}")

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Model loaded ← {path}")
