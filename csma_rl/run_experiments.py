"""
run_experiments.py -- density sweep: BEB vs fixed-CW vs analytic oracle vs DQN variants.

Protocols
  beb          802.11 binary exponential backoff (CWmin=16, CWmax=1024)
  fixed_best   best single static CW per N (tuned on separate seeds)  -> strong non-learning baseline
  analytic     CW from Bianchi's optimal tau*, knows N                 -> upper-bound reference
  dqn_indep    N independent DQNs (the paper's setting)
  dqn_shared   one DQN, parameters shared by all nodes, local observations only (parameter sharing)
  dqn_one      ONE learning node, the other N-1 use BEB (isolates non-stationarity from exploration cost)

Training is in simulated slots; evaluation is greedy (eps=0, no learning) on fresh seeds.
Usage:  python run_experiments.py --quick          (about a minute)
        python run_experiments.py                  (full sweep, 10 seeds)
"""
import argparse, csv, os, time
import numpy as np
from csmasim import (simulate, BEB, FixedCW, DQNPolicy, DQNAgent, analytic_cw, bianchi)

ACTIONS = {"wide": [8, 16, 32, 64, 128, 256, 512, 1024], "paper": [4, 8, 16, 32, 64]}
GRID = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def tcrit(df, conf=0.95):
    try:
        from scipy import stats
        return stats.t.ppf(0.5 + conf / 2, df)
    except Exception:
        return {1: 12.7, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 9: 2.26}.get(df, 1.96)


def run_one(proto, n, seed, a):
    cw_list = ACTIONS[a.actions]
    env = dict(tx_slots=a.tx_slots, lam=a.lam)
    ev = dict(slots=a.eval_slots, warmup=3000, **env)
    erng = lambda: np.random.default_rng(10_000 + seed)
    train_slots = a.train_slots
    eps = lambda t: max(0.05, 1.0 - t / (0.5 * train_slots))

    if proto == "beb":
        return simulate(n, [BEB()] * n, rng=erng(), **ev)
    if proto == "analytic":
        return simulate(n, [FixedCW(analytic_cw(n, a.tx_slots))] * n, rng=erng(), **ev)
    if proto == "fixed_best":
        best, bs = None, -1
        for cw in GRID:   # tune on separate seeds
            s = np.mean([simulate(n, [FixedCW(cw)] * n, rng=np.random.default_rng(500 + k),
                                  slots=a.eval_slots // 2, warmup=2000, **env)["throughput"] for k in range(2)])
            if s > bs:
                best, bs = cw, s
        r = simulate(n, [FixedCW(best)] * n, rng=erng(), **ev)
        r["chosen_cw"] = best
        return r

    rng = np.random.default_rng(seed)
    if proto == "dqn_indep":
        pols = [DQNPolicy(DQNAgent(len(cw_list), np.random.default_rng(seed * 1000 + i), lr=a.lr, gamma=a.gamma),
                          cw_list) for i in range(n)]
    elif proto == "dqn_shared":
        p = DQNPolicy(DQNAgent(len(cw_list), rng, lr=a.lr, gamma=a.gamma), cw_list)
        pols = [p] * n
    elif proto == "dqn_one":
        pols = [DQNPolicy(DQNAgent(len(cw_list), rng, lr=a.lr, gamma=a.gamma), cw_list)] + [BEB()] * (n - 1)
    else:
        raise ValueError(proto)
    simulate(n, pols, train_slots, rng, train=True, eps_fn=eps, reward=a.reward, beta=a.beta, **env)
    for p in {id(x): x for x in pols}.values():
        if p.learns:
            p.eps = 0.0
    return simulate(n, pols, rng=erng(), **ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 16, 20])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-slots", type=int, default=300000)
    ap.add_argument("--eval-slots", type=int, default=200000)
    ap.add_argument("--tx-slots", type=int, default=20)
    ap.add_argument("--lam", type=float, default=None, help="per-slot arrival prob per node; omit = saturated")
    ap.add_argument("--actions", choices=ACTIONS, default="wide")
    ap.add_argument("--reward", choices=["time", "paper", "team"], default="time")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--protocols", nargs="+",
                    default=["beb", "fixed_best", "analytic", "dqn_indep", "dqn_shared", "dqn_one"])
    ap.add_argument("--seed-offset", type=int, default=0, help="use e.g. 100 when tuning hyperparameters, 0 for final runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.nodes, a.seeds, a.train_slots, a.eval_slots = [2, 10, 20], 3, 60000, 60000
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for n in a.nodes:
        for proto in a.protocols:
            t0 = time.time()
            for seed in range(a.seeds):
                r = run_one(proto, n, seed + a.seed_offset, a)
                tot = r["per_node_succ"].sum()
                rows.append(dict(proto=proto, n=n, seed=seed, throughput=r["throughput"], p_coll=r["p_coll"],
                                 jain=r["jain"], delay=r["delay"], node0_share=r["per_node_succ"][0] / max(tot, 1),
                                 mean_cw=r["mean_cw"], node0_cw=r["per_node_mean_cw"][0],
                                 others_cw=(r["per_node_mean_cw"][1:].mean() if n > 1 else float("nan")),
                                 chosen_cw=r.get("chosen_cw", "")))
            print(f"N={n:2d} {proto:10s} done in {time.time()-t0:5.1f}s", flush=True)
    with open(f"{a.out}/raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary table
    metrics = ["throughput", "p_coll", "jain", "delay", "mean_cw", "node0_cw"]
    summ = {}
    for pr in a.protocols:
        for n in a.nodes:
            sel = [r for r in rows if r["proto"] == pr and r["n"] == n]
            for m in metrics:
                v = np.array([r[m] for r in sel], float)
                ci = tcrit(len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                summ[(pr, n, m)] = (v.mean(), ci)
    print("\nThroughput (mean +- 95% CI, saturated utilisation) and Bianchi check for BEB")
    print("N    " + "".join(f"{p:>16s}" for p in a.protocols) + "   bianchi(BEB)")
    for n in a.nodes:
        line = f"{n:<5d}" + "".join(f"{summ[(p, n, 'throughput')][0]:9.3f}±{summ[(p, n, 'throughput')][1]:.3f}" for p in a.protocols)
        print(line + (f"   {bianchi(n, tx_slots=a.tx_slots)[0]:.3f}" if a.lam is None else ""))
    with open(f"{a.out}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["proto", "n"] + [f"{m}_{s}" for m in metrics for s in ("mean", "ci95")])
        for p in a.protocols:
            for n in a.nodes:
                w.writerow([p, n] + [f"{summ[(p, n, m)][i]:.5f}" for m in metrics for i in (0, 1)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = {"beb": ("k", "o", "-"), "fixed_best": ("tab:blue", "s", "--"), "analytic": ("tab:gray", "^", ":"),
           "dqn_indep": ("tab:red", "v", "-"), "dqn_shared": ("tab:green", "D", "-"), "dqn_one": ("tab:orange", "x", "-.")}
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for k, (m, yl) in enumerate([("throughput", "Throughput (utilisation)"), ("p_coll", "Collision probability"),
                                 ("jain", "Jain fairness"), ("mean_cw", "Mean chosen CW")]):
        for p in a.protocols:
            c, mk, ls = sty[p]
            y = np.array([summ[(p, n, m)][0] for n in a.nodes])
            e = np.array([summ[(p, n, m)][1] for n in a.nodes])
            ax[k].errorbar(a.nodes, y, yerr=e, color=c, marker=mk, ls=ls, ms=5, capsize=2, lw=1.2, label=p)
        if m == "mean_cw":
            ax[k].set_yscale("log")
        ax[k].set_xlabel("Number of nodes N")
        ax[k].set_ylabel(yl)
        ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{a.out}/density_sweep.png", dpi=200)
    print(f"\nSaved {a.out}/raw.csv, summary.csv, density_sweep.png")


if __name__ == "__main__":
    main()
