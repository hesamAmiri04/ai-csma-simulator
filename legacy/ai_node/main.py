"""
ai_node/main.py
===============
Entry point for an AI-CSMA node container.

Each node:
  1. Maintains a packet queue (Poisson-distributed arrivals).
  2. Builds a state vector from recent channel observations.
  3. Asks the DQN agent to choose a contention window (CW).
  4. Runs one CSMA attempt via the MAC state machine.
  5. Computes the reward and trains the DQN agent.
  6. Logs per-step metrics to stdout (JSON) for later plotting.

Environment variables
---------------------
NODE_ID       : unique node identifier            (default: auto from hostname)
MEDIUM_HOST   : hostname/IP of the medium server  (default: "medium")
MEDIUM_PORT   : UDP port of the medium server     (default: 9000)
ARRIVAL_RATE  : mean packets per second (Poisson) (default: 5.0)
MAX_QUEUE     : maximum queue depth               (default: 20)
HISTORY_LEN   : sliding window length for state   (default: 10)
LOG_INTERVAL  : steps between metric log lines    (default: 50)
"""

import os
import time
import json
import random
import socket
import logging
import collections
import numpy as np

from agent    import DQNAgent
from csma_mac import CSMAMAC, CSMAResult

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("node")

# ── Configuration ─────────────────────────────────────────────────────────────
NODE_ID      = os.environ.get("NODE_ID",      socket.gethostname())
ARRIVAL_RATE = float(os.environ.get("ARRIVAL_RATE", 5.0))   # pkts/sec
MAX_QUEUE    = int(os.environ.get("MAX_QUEUE",       20))
HISTORY_LEN  = int(os.environ.get("HISTORY_LEN",    10))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL",   50))

# ── Reward weights (matching the blueprint formula) ───────────────────────────
R_SUCCESS   =  5.0
R_COLLISION = -10.0
R_BUSY      = -0.5   # per busy slot incurred


def build_state(
    sense_history:   collections.deque,   # 1=BUSY, 0=FREE per slot
    last_collision:  bool,
    queue_size:      int,
    tx_history:      collections.deque,   # 1=SUCCESS, 0=COLLISION per TX
) -> np.ndarray:
    """
    Encode the 4-dimensional state vector.

    Features
    --------
    [0] busy_ratio      – fraction of recent slots that were BUSY
    [1] collision_flag  – 1.0 if last TX was a collision, else 0.0
    [2] norm_queue      – queue depth normalised to [0, 1]
    [3] success_ratio   – fraction of recent TXs that succeeded
    """
    busy_ratio    = sum(sense_history) / max(len(sense_history), 1)
    collision_f   = 1.0 if last_collision else 0.0
    norm_queue    = min(queue_size / MAX_QUEUE, 1.0)
    success_ratio = (sum(tx_history) / max(len(tx_history), 1)
                     if tx_history else 0.5)

    return np.array([busy_ratio, collision_f, norm_queue, success_ratio],
                    dtype=np.float32)


def compute_reward(result: str, busy_slots: int) -> float:
    """
    R = w1·Throughput - w2·Collision - w3·Delay

    Simplified as:
      +5 on success, -10 on collision, -0.5 per busy slot sensed
    """
    base = R_SUCCESS if result == CSMAResult.SUCCESS else R_COLLISION
    return base + R_BUSY * busy_slots


def main():
    log.info("Node %s starting …", NODE_ID)

    # Give medium time to start (especially on first compose up)
    time.sleep(2.0)

    mac   = CSMAMAC(NODE_ID)
    agent = DQNAgent(NODE_ID)
    agent.load()   # resume from checkpoint if it exists

    # Sliding histories
    sense_history = collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)
    tx_history    = collections.deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)

    queue_size    = 0
    last_coll     = False
    step          = 0

    # Running metrics
    total_success   = 0
    total_collision = 0
    total_reward    = 0.0
    window_reward   = 0.0

    # Inter-arrival time (exponential distribution → Poisson process)
    mean_interarrival = 1.0 / ARRIVAL_RATE

    next_arrival = time.time() + random.expovariate(ARRIVAL_RATE)

    while True:
        step += 1
        now = time.time()

        # ── Packet arrivals ───────────────────────────────────────────────────
        while now >= next_arrival:
            if queue_size < MAX_QUEUE:
                queue_size += 1
            next_arrival += random.expovariate(ARRIVAL_RATE)

        if queue_size == 0:
            # Nothing to send; sleep one slot
            time.sleep(0.01)
            sense_history.append(0)
            continue

        # ── Build state ───────────────────────────────────────────────────────
        state = build_state(sense_history, last_coll, queue_size, tx_history)

        # ── Agent selects CW ──────────────────────────────────────────────────
        action = agent.select_action(state)
        cw     = agent.cw_from_action(action)

        # ── MAC transmission attempt ──────────────────────────────────────────
        result, metrics = mac.attempt_transmit(contention_window=cw)

        # ── Update histories ──────────────────────────────────────────────────
        sense_history.append(1 if metrics["busy_slots"] > 0 else 0)
        tx_history.append(1 if result == CSMAResult.SUCCESS else 0)
        last_coll = (result == CSMAResult.COLLISION)

        if result == CSMAResult.SUCCESS:
            queue_size   = max(0, queue_size - 1)
            total_success += 1
        else:
            total_collision += 1

        # ── Reward and next state ─────────────────────────────────────────────
        reward     = compute_reward(result, metrics["busy_slots"])
        next_state = build_state(sense_history, last_coll, queue_size, tx_history)

        total_reward  += reward
        window_reward += reward

        # ── Store transition and train ────────────────────────────────────────
        done = False   # continuous task; no episode termination
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.learn()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            total_tx = total_success + total_collision
            record   = {
                "node":          NODE_ID,
                "step":          step,
                "epsilon":       round(agent.epsilon, 4),
                "cw":            cw,
                "total_tx":      total_tx,
                "successes":     total_success,
                "collisions":    total_collision,
                "collision_rate": round(total_collision / max(total_tx, 1), 4),
                "throughput":    round(total_success / max(total_tx, 1), 4),
                "avg_reward":    round(total_reward / step, 4),
                "window_reward": round(window_reward / LOG_INTERVAL, 4),
                "queue":         queue_size,
                "loss":          round(loss, 6) if loss is not None else None,
            }
            print(json.dumps(record), flush=True)
            window_reward = 0.0

        # Save checkpoint every 1000 steps
        if step % 1000 == 0:
            agent.save()
            log.info("[%s] checkpoint saved at step %d", NODE_ID, step)


if __name__ == "__main__":
    main()
