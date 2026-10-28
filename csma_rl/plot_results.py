"""plot_results.py -- redraw the density-sweep figure from a results folder's summary.csv
Usage: python3 plot_results.py results_team
"""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = list(csv.DictReader(open(os.path.join(d, "summary.csv"))))
protos = list(dict.fromkeys(r["proto"] for r in rows))
sty = {"beb": ("k", "o", "-"), "fixed_best": ("tab:blue", "s", "--"), "analytic": ("tab:gray", "^", ":"),
       "dqn_indep": ("tab:red", "v", "-"), "dqn_shared": ("tab:green", "D", "-"), "dqn_one": ("tab:orange", "x", "-.")}
panels = [("throughput", "Throughput (utilisation)"), ("p_coll", "Collision probability"),
          ("jain", "Jain fairness"), ("mean_cw", "Mean chosen CW")]
panels = [p for p in panels if f"{p[0]}_mean" in rows[0]]
fig, ax = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.6))
for k, (m, yl) in enumerate(panels):
    for p in protos:
        sel = sorted([r for r in rows if r["proto"] == p], key=lambda r: int(r["n"]))
        c, mk, ls = sty.get(p, ("m", "o", "-"))
        ax[k].errorbar([int(r["n"]) for r in sel], [float(r[f"{m}_mean"]) for r in sel],
                       yerr=[float(r[f"{m}_ci95"]) for r in sel], color=c, marker=mk, ls=ls, ms=5, capsize=2, lw=1.2, label=p)
    if m == "mean_cw":
        ax[k].set_yscale("log")
    ax[k].set_xlabel("Number of nodes N"); ax[k].set_ylabel(yl); ax[k].grid(alpha=0.3)
ax[0].legend(fontsize=7)
plt.tight_layout()
out = os.path.join(d, "density_sweep.png")
plt.savefig(out, dpi=200)
print("Saved", out)
