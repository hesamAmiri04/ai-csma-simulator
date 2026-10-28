"""
csmasim.py -- clean slotted CSMA/CA simulator for comparing contention-window policies.

Model (Bianchi-style virtual slots, single collision domain, perfect carrier sense)
  * An "event" is either an idle slot (duration 1) or a busy period of `tx_slots`
    slots (success or collision last the same time, as in Bianchi's Ts = Tc).
  * Counter-based backoff with FREEZING: every contending node draws a counter
    uniformly from {0..CW-1}; the counter decrements only in idle slots; a node
    transmits when its counter is 0. Nodes whose counters are not 0 are frozen
    during busy periods.
  * Exactly one transmitter => success; two or more => all of them collide.
  * Traffic: saturated (lam=None) or Bernoulli arrivals per slot with a finite queue.
  * A new CW is chosen by the node's policy each time it starts a backoff
    (new packet or retransmission), from LOCAL information only.

Metrics (measured only after `warmup`):
  throughput  = successful_transmissions * tx_slots / elapsed_slots  (channel utilisation)
  p_coll      = collided transmission attempts / all transmission attempts (per node)
  jain        = Jain fairness over per-node success counts
  delay       = mean head-of-line-to-success time, in slots
"""
import math
import collections
import numpy as np

OBS_DIM = 5


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
class BEB:
    """802.11 binary exponential backoff: CWmin after success, double after collision."""
    learns = False

    def __init__(self, cw_min=16, cw_max=1024):
        self.cw_min, self.cw_max = cw_min, cw_max

    def choose(self, obs, last_cw, last_success):
        if last_success is None or last_success:
            return self.cw_min, None
        return min(2 * last_cw, self.cw_max), None


class FixedCW:
    learns = False

    def __init__(self, cw):
        self.cw = int(cw)

    def choose(self, obs, last_cw, last_success):
        return self.cw, None


def analytic_cw(n, tx_slots):
    """Oracle that knows N: tau* ~ 1/(N*sqrt(Tc*/2)) (Bianchi, Tc* = tx_slots), W = 2/tau - 1."""
    tau = 1.0 / (n * math.sqrt(tx_slots / 2.0))
    return max(2, int(round(2.0 / tau - 1.0)))


class DQNPolicy:
    learns = True

    def __init__(self, agent, cw_list):
        self.agent, self.cw_list = agent, list(cw_list)
        self.eps = 0.0

    def choose(self, obs, last_cw, last_success):
        a = self.agent.act(obs, self.eps)
        return self.cw_list[a], a

    def observe(self, s, a, r, s2):
        self.agent.push(s, a, r, s2)


# --------------------------------------------------------------------------- #
# NumPy DQN (no torch dependency)
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, math.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        a = x
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            a = z if l == L - 1 else np.maximum(z, 0.0)
            acts.append(a)
        return a, acts

    def backward(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for l in reversed(range(len(self.W))):
            gW[l] = acts[l].T @ d
            gb[l] = d.sum(0)
            if l > 0:
                d = (d @ self.W[l].T) * (acts[l] > 0)
        return gW, gb

    def adam(self, gW, gb, lr, clip=10.0, b1=0.9, b2=0.999, eps=1e-8):
        norm = math.sqrt(sum((g ** 2).sum() for g in gW + gb))
        sc = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for l in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                gl = g[l] * sc
                m[l] = b1 * m[l] + (1 - b1) * gl
                v[l] = b2 * v[l] + (1 - b2) * gl * gl
                mh = m[l] / (1 - b1 ** self.t)
                vh = v[l] / (1 - b2 ** self.t)
                p[l] -= lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, o):
        self.W = [w.copy() for w in o.W]
        self.b = [b.copy() for b in o.b]


class DQNAgent:
    def __init__(self, n_actions, rng, obs_dim=OBS_DIM, hidden=64, lr=1e-3, gamma=0.9,
                 buffer=20000, batch=32, target_sync=100, learn_start=200):
        self.rng, self.nA, self.gamma, self.lr = rng, n_actions, gamma, lr
        self.batch, self.target_sync, self.learn_start = batch, target_sync, learn_start
        sizes = [obs_dim, hidden, hidden, n_actions]
        self.net, self.tgt = MLP(sizes, rng), MLP(sizes, rng)
        self.tgt.copy_from(self.net)
        self.S = np.zeros((buffer, obs_dim), np.float32)
        self.S2 = np.zeros((buffer, obs_dim), np.float32)
        self.A = np.zeros(buffer, np.int64)
        self.R = np.zeros(buffer, np.float32)
        self.cap, self.size, self.ptr, self.grad_steps = buffer, 0, 0, 0

    def act(self, obs, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        q, _ = self.net.forward(obs[None, :])
        return int(q[0].argmax())

    def push(self, s, a, r, s2):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i] = s, a, r, s2
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def learn(self):
        if self.size < self.learn_start:
            return
        idx = self.rng.integers(0, self.size, self.batch)
        s, a, r, s2 = self.S[idx], self.A[idx], self.R[idx], self.S2[idx]
        qn, _ = self.tgt.forward(s2)
        target = r + self.gamma * qn.max(1)
        q, acts = self.net.forward(s)
        err = q[np.arange(self.batch), a] - target
        d = np.zeros_like(q)
        d[np.arange(self.batch), a] = np.clip(err, -1.0, 1.0) / self.batch   # Huber gradient
        gW, gb = self.net.backward(acts, d)
        self.net.adam(gW, gb, self.lr)
        self.grad_steps += 1
        if self.grad_steps % self.target_sync == 0:       # synced every `target_sync` GRADIENT steps
            self.tgt.copy_from(self.net)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class _Node:
    def __init__(self, hist_len, cw_init):
        self.queue = 0
        self.counter = None            # None = not contending
        self.last_cw = cw_init
        self.last_success = None
        self.busy_hist = collections.deque([0] * hist_len, maxlen=hist_len)
        self.tx_hist = collections.deque(maxlen=hist_len)
        self.pending = None            # (obs, action, t0, busy_events, global_successes_at_t0)
        self.hol_start = 0


def simulate(n, policies, slots, rng, tx_slots=20, lam=None, queue_cap=50, train=False,
             eps_fn=None, reward="time", beta=0.1, warmup=0, hist_len=10, cw_max=1024, freeze=True):
    """Run `slots` slots (after `warmup`). `policies` is a list of n policy objects
    (the same object may appear several times, e.g. a shared DQN). Returns a metrics dict."""
    nodes = [_Node(hist_len, 16) for _ in range(n)]
    sat = lam is None
    log_cwmax = math.log2(cw_max)
    learners = {id(p): p for p in policies if p.learns}.values()
    gs_cum = 0                       # global successful transmissions so far (for the team reward)
    cw_sum, cw_cnt = np.zeros(n), np.zeros(n)   # chosen CWs during the measurement window

    def obs_of(nd):
        return np.array([sum(nd.busy_hist) / hist_len,
                         1.0 if nd.last_success is False else 0.0,
                         1.0 if sat else min(nd.queue / queue_cap, 1.0),
                         (sum(nd.tx_hist) / len(nd.tx_hist)) if nd.tx_hist else 0.5,
                         math.log2(nd.last_cw) / log_cwmax], np.float32)

    def start_backoff(i, t):
        nd = nodes[i]
        o = obs_of(nd)
        cw, a = policies[i].choose(o, nd.last_cw, nd.last_success)
        nd.last_cw = cw
        nd.counter = int(rng.integers(cw))
        nd.pending = (o, a, t, 0, gs_cum)
        if t >= warmup:
            cw_sum[i] += cw
            cw_cnt[i] += 1

    for i in range(n):
        if sat:
            nodes[i].queue = 1
        start_backoff(i, 0)

    total = slots + warmup
    t = 0
    succ = att = coll = 0
    per_succ = np.zeros(n)
    dsum = dn = 0.0
    t_meas0 = None

    while t < total:
        if t_meas0 is None and t >= warmup:
            t_meas0 = t
        if train and eps_fn is not None:
            e = eps_fn(t)
            for p in learners:
                p.eps = e
        ready = [i for i in range(n) if nodes[i].counter == 0]
        if not ready:
            dur, busy = 1, 0
            for nd in nodes:
                if nd.counter is not None:
                    nd.counter -= 1
        else:
            dur, busy = tx_slots, 1
            rs = set(ready)
            for i, nd in enumerate(nodes):                       # frozen nodes see a busy period
                if nd.counter is not None and i not in rs and nd.pending is not None:
                    o, a, t0, b, g0 = nd.pending
                    nd.pending = (o, a, t0, b + 1, g0)
                    if not freeze:                               # validation only: no freezing
                        nd.counter -= 1
        t_end = t + dur
        for nd in nodes:
            nd.busy_hist.append(busy)

        if ready:
            ok = len(ready) == 1
            if ok:
                gs_cum += 1
            measuring = t >= (t_meas0 if t_meas0 is not None else warmup)
            for i in ready:
                nd = nodes[i]
                o, a, t0, busy_ev, g0 = nd.pending
                nd.tx_hist.append(1 if ok else 0)
                nd.last_success = ok
                if measuring:
                    att += 1
                    if ok:
                        succ += 1
                        per_succ[i] += 1
                        dsum += t_end - nd.hol_start
                        dn += 1
                    else:
                        coll += 1
                if ok:
                    nd.queue -= 1 if not sat else 0
                    if sat:
                        nd.hol_start = t_end
                # transition for learners
                if a is not None and train:
                    if reward == "paper":
                        r = (5.0 if ok else -10.0) - 0.5 * busy_ev
                    elif reward == "team":      # SYSTEM utilisation over this node's decision interval
                        r = 5.0 * (gs_cum - g0) * tx_slots / max(t_end - t0, 1)
                    else:
                        r = (1.0 if ok else -1.0) - beta * (t_end - t0) / tx_slots
                    policies[i].observe(o, a, r, obs_of(nd))
                    policies[i].agent.learn()
                nd.counter = None
                nd.pending = None
                if nd.queue > 0 or sat:
                    start_backoff(i, t_end)
        t = t_end
        if not sat:
            pa = 1.0 - (1.0 - lam) ** dur
            for i, nd in enumerate(nodes):
                if rng.random() < pa and nd.queue < queue_cap:
                    nd.queue += 1
                    if nd.counter is None:
                        nd.hol_start = t
                        start_backoff(i, t)

    el = max(t - (t_meas0 if t_meas0 is not None else 0), 1)
    x = per_succ
    return {
        "throughput": succ * tx_slots / el,
        "p_coll": coll / max(att, 1),
        "jain": float(x.sum() ** 2 / (n * (x ** 2).sum())) if x.sum() > 0 else 0.0,
        "delay": dsum / max(dn, 1),
        "mean_cw": float(cw_sum.sum() / max(cw_cnt.sum(), 1)),
        "per_node_mean_cw": cw_sum / np.maximum(cw_cnt, 1),
        "per_node_succ": per_succ,
    }


# --------------------------------------------------------------------------- #
# Analytical check (Bianchi 2000), saturated, Ts = Tc = tx_slots
# --------------------------------------------------------------------------- #
def bianchi(n, w=16, m=6, tx_slots=20, iters=2000):
    tau = 0.1
    for _ in range(iters):
        p = 1 - (1 - tau) ** (n - 1)
        new = 2 * (1 - 2 * p) / ((1 - 2 * p) * (w + 1) + p * w * (1 - (2 * p) ** m)) if abs(1 - 2 * p) > 1e-9 else tau
        tau = 0.5 * tau + 0.5 * new
    p = 1 - (1 - tau) ** (n - 1)
    ptr = 1 - (1 - tau) ** n
    ps = n * tau * (1 - tau) ** (n - 1) / ptr
    s = ps * ptr * tx_slots / ((1 - ptr) + ptr * tx_slots)
    return s, p
