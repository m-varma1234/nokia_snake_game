"""
A2C Agent — Advantage Actor-Critic (synchronous, n-step returns).
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


class A2CAgent:
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 20,
        device: str | None = None,
    ):
        self.gamma        = gamma
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps      = n_steps

        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else "cpu"
            )
        )

        self.net       = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, eps=1e-5)

        self._clear_rollout()

    def _clear_rollout(self):
        self._states  = []
        self._actions = []
        self._rewards = []
        self._dones   = []
        self._values  = []
        self._log_probs  = []
        self._entropies  = []

    # ------------------------------------------------------------------ act
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        dist   = Categorical(logits=logits)
        action = dist.sample()

        if training:
            self._states.append(state)
            self._actions.append(action.item())
            self._values.append(value.item())
            self._log_probs.append(dist.log_prob(action))
            self._entropies.append(dist.entropy())

        return action.item()

    def store_reward(self, reward: float, done: bool):
        self._rewards.append(reward)
        self._dones.append(float(done))

    # ----------------------------------------------------------- n-step learn
    def learn(self, last_state: np.ndarray | None = None) -> float | None:
        if len(self._rewards) < self.n_steps:
            return None

        # Bootstrap from last state
        if last_state is not None:
            s = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, last_val = self.net(s)
            last_val = last_val.item()
        else:
            last_val = 0.0

        # Compute n-step returns
        returns = []
        R = last_val
        for r, d in zip(reversed(self._rewards), reversed(self._dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t  = torch.tensor(self._values, dtype=torch.float32, device=self.device)
        adv_t     = (returns_t - values_t).detach()
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        log_probs_t = torch.stack(self._log_probs)
        entropies_t = torch.stack(self._entropies)

        # Re-compute values for critic loss (use stored values to avoid graph issues)
        states_t = torch.tensor(np.array(self._states), dtype=torch.float32, device=self.device)
        _, values_pred = self.net(states_t)

        actor_loss  = -(log_probs_t * adv_t).mean()
        critic_loss = nn.functional.mse_loss(values_pred, returns_t)
        entropy     = entropies_t.mean()
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        self._clear_rollout()
        return loss.item()

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
