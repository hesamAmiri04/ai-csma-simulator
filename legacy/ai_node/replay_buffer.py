"""
ai_node/replay_buffer.py
========================
Fixed-size experience replay buffer for DQN training.

Each transition is stored as:
    (state, action, reward, next_state, done)
where state/next_state are 1-D numpy arrays.
"""

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Uniform-random experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 10_000):
        self._buf = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch.

        Returns
        -------
        Tuple of (states, actions, rewards, next_states, dones)
        each as a numpy array ready for conversion to tensors.
        """
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)
