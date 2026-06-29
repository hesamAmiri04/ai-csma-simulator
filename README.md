# AI-CSMA Simulator 🧠📡

A Dockerized simulation of an **AI-enhanced CSMA (Carrier Sense Multiple Access)** protocol where each wireless node uses a **Deep Q-Network (DQN)** to dynamically choose its contention window — replacing the rigid binary exponential backoff of standard 802.11 CSMA/CA.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network (csma_net)               │
│                                                             │
│   ┌──────────────┐          UDP          ┌──────────────┐  │
│   │   ai_node_1  │ ◄──────────────────► │              │  │
│   │   (DQN)      │                       │   medium     │  │
│   ├──────────────┤    SENSE / TX_START   │  (Virtual    │  │
│   │   ai_node_2  │ ◄──────────────────► │   Air)       │  │
│   │   (DQN)      │    FREE / ACK /       │              │  │
│   ├──────────────┤    COLLISION          └──────────────┘  │
│   │   ai_node_N  │                                         │
│   │   (DQN)      │                                         │
│   └──────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

Each `ai_node` container is an **independent wireless device** with its own DQN model. They share no parameters — each node learns the optimal backoff policy from its own channel observations. The `medium` container is the **virtual air**: it arbitrates transmissions and detects collisions.

---

## Repository Structure

```
ai-csma-simulator/
├── README.md
├── docker-compose.yml          # Orchestrates medium + N ai_node containers
├── compare.py                  # Local simulation: AI-CSMA vs 802.11 (generates plot)
├── requirements.txt            # Local deps for compare.py
│
├── wireless_medium/
│   ├── medium.py               # UDP channel server (SENSE/TX/ACK/COLLISION)
│   └── Dockerfile
│
└── ai_node/
    ├── agent.py                # DQN agent (QNetwork, epsilon-greedy, Bellman update)
    ├── replay_buffer.py        # Fixed-size uniform experience replay
    ├── csma_mac.py             # CSMA state machine (SENSE→BACKOFF→TX→ACK)
    ├── main.py                 # Node main loop, state builder, reward computation
    └── Dockerfile
```

---

## The AI-CSMA Protocol

### How It Differs from Standard CSMA/CA

| Feature               | Standard 802.11 CSMA/CA         | AI-CSMA (DQN)                           |
|-----------------------|----------------------------------|-----------------------------------------|
| Backoff logic         | Binary exponential (fixed rules) | DQN chooses CW from {4, 8, 16, 32, 64} |
| Adaptation speed      | Reactive (doubles on collision)  | Predictive (learns traffic patterns)    |
| Collision recovery    | Doubles CW blindly               | Adjusts CW based on channel history     |
| Dense networks        | Throughput collapses             | Maintains stable throughput             |

### State Vector (4 features)

$$\mathbf{s} = \bigl[\underbrace{B_r}_{\text{busy ratio}},\; \underbrace{C_f}_{\text{collision flag}},\; \underbrace{Q/Q_{max}}_{\text{norm. queue}},\; \underbrace{S_r}_{\text{success ratio}}\bigr]$$

| Feature | Description |
|---------|-------------|
| `busy_ratio` | Fraction of last 10 slots the channel was sensed BUSY |
| `collision_flag` | 1 if the previous transmission resulted in a collision |
| `norm_queue` | Current queue depth normalised to [0, 1] |
| `success_ratio` | Fraction of last 10 transmissions that succeeded |

### Action Space

| Action | Contention Window |
|--------|-----------------|
| 0 | CW = 4  (very aggressive) |
| 1 | CW = 8 |
| 2 | CW = 16  (802.11 default) |
| 3 | CW = 32 |
| 4 | CW = 64  (conservative) |

### Reward Function

$$R = \begin{cases}+5 & \text{successful ACK} \\ -10 & \text{collision} \end{cases} \;-\; 0.5 \times \text{busy\_slots}$$

The large collision penalty encourages the agent to select larger CW values when the channel is congested, while the busy-slot penalty discourages unnecessary waiting.

### DQN Architecture

```
Input (4)  →  Linear(128)  →  ReLU  →  Linear(128)  →  ReLU  →  Output (5)
```

Training details:
- **Experience replay**: 10,000-transition buffer, sampled uniformly
- **Target network**: synced every 100 gradient steps (stabilises training)
- **Optimiser**: Adam, lr = 3×10⁻⁴
- **Loss**: MSE on Bellman targets — $Q(s,a) \leftarrow r + \gamma \max_{a'} Q_{\text{target}}(s', a')$
- **Exploration**: ε-greedy, ε decays from 1.0 → 0.05 over 500 steps

---

## Quick Start

### 1. Run with Docker Compose (5 nodes)

```bash
git clone https://github.com/your-username/ai-csma-simulator
cd ai-csma-simulator
docker compose up --build
```

### 2. Scale to N nodes (density experiment)

```bash
docker compose up --build --scale ai_node=10
```

### 3. Generate the comparison plot (locally, no Docker)

```bash
pip install numpy matplotlib
python compare.py
# → saves comparison_plot.png
```

---

## Environment Variables

### wireless_medium

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIUM_PORT` | `9000` | UDP port to listen on |
| `SLOT_DURATION_MS` | `10` | Logical slot duration |

### ai_node

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIUM_HOST` | `medium` | Hostname of the medium container |
| `MEDIUM_PORT` | `9000` | UDP port of the medium |
| `ARRIVAL_RATE` | `5.0` | Packet arrivals per second (Poisson) |
| `MAX_QUEUE` | `20` | Maximum packet queue depth |
| `HISTORY_LEN` | `10` | Sliding window for state features |
| `DQN_LR` | `3e-4` | Adam learning rate |
| `DQN_GAMMA` | `0.99` | Discount factor γ |
| `DQN_EPS_DECAY` | `500` | ε-greedy decay constant |
| `DQN_BATCH` | `64` | Mini-batch size |
| `DQN_TARGET_UPDATE` | `100` | Target network sync interval |

---

## UDP Protocol

```
Node → Medium:
  SENSE <node_id>                   → Medium replies FREE or BUSY
  TX_START <node_id> <duration_ms>  → begin a transmission
  TX_END   <node_id>                → (optional) early end

Medium → Node:
  FREE                              → channel is idle
  BUSY                              → channel is occupied
  ACK  <node_id>                    → delivery succeeded
  COLLISION <node_id>               → collision detected
```

---

## Results

Running `python compare.py` generates `comparison_plot.png` showing throughput and collision rate as a function of node count. Expected outcome:

- **Standard 802.11**: throughput degrades sharply beyond ~8 nodes due to BEB oscillation
- **AI-CSMA**: the DQN agent learns to select larger CW values proactively, maintaining stable throughput even at 20 nodes

---

## License

MIT
