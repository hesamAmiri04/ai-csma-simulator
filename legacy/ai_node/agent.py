"""
ai_node/agent.py
================
Deep Q-Network (DQN) agent for the AI-CSMA protocol.

State space  (4 features):
  [0] busy_ratio      – fraction of last N slots the channel was sensed BUSY   ∈ [0, 1]
  [1] collision_flag  – 1 if the last transmission resulted in a collision      ∈ {0, 1}
  [2] norm_queue      – normalized queue length  (queue / MAX_QUEUE)            ∈ [0, 1]
  [3] success_ratio   – fraction of last N transmissions that succeeded         ∈ [0, 1]

Action space (5 discrete actions → contention window sizes):
  0 → CW =  4   (very aggressive)
  1 → CW =  8
  2 → CW = 16   (802.11 default initial CW)
  3 → CW = 32
  4 → CW = 64   (conservative)

Reward function:
  R = +5   on successful ACK
  R = -10  on COLLISION
  R = -0.5 per slot spent sensing a BUSY channel (delay penalty)

Network architecture:
  Linear(4, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 5)

Training:
  • Experience replay (ReplayBuffer)
  • Separate target network, synced every TARGET_UPDATE steps
  • Adam optimiser, MSE loss on Bellman targets
  • ε-greedy exploration with exponential decay
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer

# ── Hyper-parameters (overridable via environment variables) ──────────────────
STATE_DIM      = 4
ACTION_DIM     = 5
CW_SIZES       = [4, 8, 16, 32, 64]          # contention window per action

LR             = float(os.environ.get("DQN_LR",           3e-4))
GAMMA          = float(os.environ.get("DQN_GAMMA",        0.99))
EPSILON_START  = float(os.environ.get("DQN_EPS_START",    1.0))
EPSILON_END    = float(os.environ.get("DQN_EPS_END",      0.05))
EPSILON_DECAY  = float(os.environ.get("DQN_EPS_DECAY",    500))   # steps
BATCH_SIZE     = int(os.environ.get("DQN_BATCH",          64))
BUFFER_SIZE    = int(os.environ.get("DQN_BUFFER",         10_000))
TARGET_UPDATE  = int(os.environ.get("DQN_TARGET_UPDATE",  100))   # steps


# ── Neural network ────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    DQN agent that learns the optimal contention window policy.

    Usage
    -----
    >>> agent = DQNAgent(node_id="node_1")
    >>> action_idx = agent.select_action(state)
    >>> cw = agent.cw_from_action(action_idx)
    >>> # … run CSMA with that CW, observe reward and next_state …
    >>> agent.store_transition(state, action_idx, reward, next_state, done)
    >>> agent.learn()
    """

    def __init__(self, node_id: str = "node"):
        self.node_id = node_id
        self.device  = torch.device("cpu")   # CPU-only inside Docker

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)
        self.loss_fn   = nn.MSELoss()

        self.steps_done    = 0
        self.episode_steps = 0    # resets each episode/epoch

    # ── Epsilon schedule ──────────────────────────────────────────────────────
    @property
    def epsilon(self) -> float:
        return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
            -self.steps_done / EPSILON_DECAY
        )

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection.  Returns action index (0-4)."""
        self.steps_done += 1
        if np.random.rand() < self.epsilon:
            return np.random.randint(ACTION_DIM)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(s)
            return int(q_vals.argmax(dim=1).item())

    def cw_from_action(self, action_idx: int) -> int:
        """Map action index to actual contention window size."""
        return CW_SIZES[action_idx]

    # ── Replay buffer ─────────────────────────────────────────────────────────
    def store_transition(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────────────────────
    def learn(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient descent step.
        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states      = torch.tensor(states,      device=self.device)
        actions     = torch.tensor(actions,     device=self.device)
        rewards     = torch.tensor(rewards,     device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones       = torch.tensor(dones,       device=self.device)

        # Current Q-values: Q(s, a)
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ · max_a' Q_target(s', a')  (zero if terminal)
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1).values
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilise training
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Sync target network periodically
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "/app/model.pt") -> None:
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "steps":  self.steps_done,
        }, path)

    def load(self, path: str = "/app/model.pt") -> None:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt["policy"])
            self.target_net.load_state_dict(ckpt["target"])
            self.steps_done = ckpt.get("steps", 0)
