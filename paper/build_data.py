import csv, sys, collections, json
import numpy as np
from scipy import stats
sys.path.insert(0, '/mnt/user-data/outputs/csma_rl')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csmasim import simulate, BEB, FixedCW, bianchi

OUT = '/mnt/user-data/outputs/paper'
SRC = {'b01': '/mnt/user-data/uploads/raw_1.csv',          # local reward, beta=0.1 (first full sweep)
       'b001': '/mnt/user-data/uploads/raw_47d583.csv',    # local reward, beta=0.01
       'team': '/mnt/user-data/uploads/raw_cfe2bf.csv'}    # team reward
NS = [2, 4, 6, 8, 10, 12, 16, 20]
def load(p):
    D = collections.defaultdict(dict)
    for r in csv.DictReader(open(p)):
        D[(r['proto'], int(r['n']))][int(r['seed'])] = r
    return D
R = {k: load(v) for k, v in SRC.items()}
def arr(k, proto, n, m):
    d = R[k][(proto, n)]
    return np.array([float(d[s][m]) for s in sorted(d)])
def ms(k, proto, n, m):
    x = arr(k, proto, n, m); t = stats.t.ppf(0.975, len(x) - 1)
    return x.mean(), t * x.std(ddof=1) / np.sqrt(len(x))
def paired(k, proto, ref, n, m='throughput'):
    d = arr(k, proto, n, m) - arr(k, ref, n, m)
    return d.mean(), stats.ttest_1samp(d, 0).pvalue

V = {}
# ---------- validation ----------
rng = lambda s: np.random.default_rng(s)
val = []
for n in [2, 5, 10, 20]:
    r = [simulate(n, [BEB()] * n, 200000, rng(s), warmup=5000) for s in range(3)]
    S = np.mean([x['throughput'] for x in r]); P = np.mean([x['p_coll'] for x in r])
    bs, bp = bianchi(n)
    val.append((n, S, bs, P, bp))
fz = []
for W in [16, 64, 128]:
    n = 10; tau = 2 / (W + 1); ptr = 1 - (1 - tau) ** n; ps = n * tau * (1 - tau) ** (n - 1) / ptr
    an = ps * ptr * 20 / ((1 - ptr) + ptr * 20)
    nf = np.mean([simulate(n, [FixedCW(W)] * n, 300000, rng(s), warmup=3000, freeze=False)['throughput'] for s in range(3)])
    wf = np.mean([simulate(n, [FixedCW(W)] * n, 300000, rng(s), warmup=3000)['throughput'] for s in range(3)])
    fz.append((W, an, nf, wf))
with open(f'{OUT}/tab_valid.tex', 'w') as f:
    f.write("\\begin{tabular}{@{}rcccccc@{}}\n\\toprule\n$N$ & $S$ sim & $S$ Bianchi & rel.\\ err. & $p_c$ sim & $p_c$ Bianchi \\\\\n\\midrule\n")
    for n, S, bs, P, bp in val:
        f.write(f"{n} & {S:.3f} & {bs:.3f} & {100*(S-bs)/bs:+.1f}\\% & {P:.3f} & {bp:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
V['val'] = val; V['fz'] = fz

# ---------- main throughput table ----------
cols = [('beb', 'team', 'BEB'), ('fixed_best', 'team', 'Fixed'), ('analytic', 'team', 'Analyt.'),
        ('dqn_indep', 'b01', None), ('dqn_indep', 'b001', None), ('dqn_indep', 'team', None),
        ('dqn_shared', 'b01', None), ('dqn_shared', 'b001', None), ('dqn_shared', 'team', None)]
with open(f'{OUT}/tab_main.tex', 'w') as f:
    f.write("\\begin{tabular}{@{}r|ccc|ccc|ccc@{}}\n\\toprule\n")
    f.write(" & \\multicolumn{3}{c|}{Baselines} & \\multicolumn{3}{c|}{Independent DQN} & \\multicolumn{3}{c}{Shared-parameter DQN} \\\\\n")
    f.write("$N$ & BEB & Fixed & Analyt. & $\\beta{=}0.1$ & $\\beta{=}0.01$ & Team & $\\beta{=}0.1$ & $\\beta{=}0.01$ & Team \\\\\n\\midrule\n")
    for n in NS:
        cells = []
        for proto, k, _ in cols:
            m, c = ms(k, proto, n, 'throughput'); cells.append(f"{m:.3f}$\\pm${c:.3f}")
        f.write(f"{n} & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- team-reward fairness / paired tests ----------
with open(f'{OUT}/tab_team.tex', 'w') as f:
    f.write("\\begin{tabular}{@{}rcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Independent DQN} & \\multicolumn{3}{c}{Shared-parameter DQN} \\\\\n")
    f.write("$N$ & $\\Delta S$ vs BEB & $p$ & Jain & $\\Delta S$ vs BEB & $p$ & Jain \\\\\n\\midrule\n")
    pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    for n in NS:
        d1, p1 = paired('team', 'dqn_indep', 'beb', n); d2, p2 = paired('team', 'dqn_shared', 'beb', n)
        j1 = ms('team', 'dqn_indep', n, 'jain')[0]; j2 = ms('team', 'dqn_shared', n, 'jain')[0]
        f.write(f"{n} & {d1:+.3f} & {pf(p1)} & {j1:.2f} & {d2:+.3f} & {pf(p2)} & {j2:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- numbers for the text ----------
g = lambda k, p, n, m='throughput': ms(k, p, n, m)[0]
for n in [2, 10, 20]:
    V[f'beb{n}'] = g('team', 'beb', n); V[f'fix{n}'] = g('team', 'fixed_best', n); V[f'ana{n}'] = g('team', 'analytic', n)
    V[f'ti{n}'] = g('team', 'dqn_indep', n); V[f'ts{n}'] = g('team', 'dqn_shared', n)
    V[f'jti{n}'] = g('team', 'dqn_indep', n, 'jain'); V[f'jts{n}'] = g('team', 'dqn_shared', n, 'jain')
for k in ['b01', 'b001']:
    for n in [2, 10, 20]:
        V[f'{k}i{n}'] = g(k, 'dqn_indep', n); V[f'{k}s{n}'] = g(k, 'dqn_shared', n)
V['b01i20pc'] = g('b01', 'dqn_indep', 20, 'p_coll'); V['b01s20pc'] = g('b01', 'dqn_shared', 20, 'p_coll'); V['beb20pc'] = g('team', 'beb', 20, 'p_coll')
V['b01i20jain'] = g('b01', 'dqn_indep', 20, 'jain')
V['fixcw'] = {n: collections.Counter(R['team'][('fixed_best', n)][s]['chosen_cw'] for s in R['team'][('fixed_best', n)]).most_common(1)[0][0] for n in [2, 10, 20]}
V['pair'] = {f'{k}_{p}_{n}': paired(k, p, 'beb', n) for k in ['team', 'b001'] for p in ['dqn_indep', 'dqn_shared'] for n in NS}
V['pairfix'] = {f'team_{p}_{n}': paired('team', p, 'fixed_best', n) for p in ['dqn_indep', 'dqn_shared'] for n in NS}
V['share'] = {k: {n: (arr(k, 'dqn_one', n, 'node0_share') * n).mean() for n in [4, 10, 20]} for k in ['b01', 'b001', 'team']}
V['cw20'] = {p: g('team', p, 20, 'mean_cw') for p in ['beb', 'fixed_best', 'analytic', 'dqn_indep', 'dqn_shared']}
V['cw20_b001'] = {p: g('b001', p, 20, 'mean_cw') for p in ['dqn_indep', 'dqn_shared']}
json.dump({k: v for k, v in V.items()}, open('V.json', 'w'), default=float, indent=1)

# ---------- figures ----------
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'lines.linewidth': 1.0, 'lines.markersize': 3.5})
def line(ax, k, proto, m, **kw):
    y = [ms(k, proto, n, m)[0] for n in NS]; e = [ms(k, proto, n, m)[1] for n in NS]
    ax.errorbar(NS, y, yerr=e, capsize=1.5, elinewidth=0.6, **kw)
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.25), sharey=True)
for a, (k, ttl) in zip(ax, [('b01', r'(a) Local reward, $\beta=0.1$'), ('b001', r'(b) Local reward, $\beta=0.01$'), ('team', '(c) Team reward')]):
    line(a, 'team', 'beb', 'throughput', color='k', marker='o', ls='-', label='BEB')
    line(a, 'team', 'fixed_best', 'throughput', color='tab:blue', marker='s', ls='--', label='Best fixed CW')
    line(a, k, 'dqn_indep', 'throughput', color='tab:red', marker='v', ls='-', label='DQN independent')
    line(a, k, 'dqn_shared', 'throughput', color='tab:green', marker='D', ls='-.', label='DQN shared')
    a.set_title(ttl, fontsize=7.5); a.set_xlabel('Number of nodes $N$'); a.grid(alpha=0.3)
ax[0].set_ylabel('Throughput (utilisation)'); ax[0].legend(loc='lower left')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_throughput.pdf'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.25))
a = ax[0]
line(a, 'team', 'beb', 'jain', color='k', marker='o', ls='-', label='BEB')
line(a, 'team', 'dqn_indep', 'jain', color='tab:red', marker='v', ls='-', label='DQN independent')
line(a, 'team', 'dqn_shared', 'jain', color='tab:green', marker='D', ls='-.', label='DQN shared')
a.set_ylabel('Jain fairness index'); a.set_title('(a) Fairness, team reward', fontsize=7.5); a.legend(loc='lower left'); a.set_ylim(0, 1.05)
a = ax[1]
line(a, 'team', 'beb', 'mean_cw', color='k', marker='o', ls='-', label='BEB')
line(a, 'team', 'analytic', 'mean_cw', color='tab:gray', marker='^', ls=':', label='Analytic $W^*$')
line(a, 'team', 'dqn_indep', 'mean_cw', color='tab:red', marker='v', ls='-', label='DQN independent')
line(a, 'team', 'dqn_shared', 'mean_cw', color='tab:green', marker='D', ls='-.', label='DQN shared')
a.set_yscale('log'); a.set_ylabel('Mean chosen CW'); a.set_title('(b) Chosen CW, team reward', fontsize=7.5); a.legend(loc='upper left')
a = ax[2]
for k, lab, c, mk in [('b01', r'local, $\beta=0.1$', 'tab:red', 'v'), ('b001', r'local, $\beta=0.01$', 'tab:orange', 's'), ('team', 'team', 'tab:green', 'D')]:
    ns = [4, 10, 20]
    y = [V['share'][k][n] for n in ns]
    e = [stats.t.ppf(0.975, 9) * (arr(k, 'dqn_one', n, 'node0_share') * n).std(ddof=1) / np.sqrt(10) for n in ns]
    a.errorbar(ns, y, yerr=e, color=c, marker=mk, capsize=1.5, elinewidth=0.6, label=lab)
a.axhline(1.0, color='k', lw=0.6, ls=':'); a.set_ylabel('Learner share / fair share'); a.set_title('(c) One learner among BEB nodes', fontsize=7.5); a.legend(loc='upper right')
for a in ax:
    a.set_xlabel('Number of nodes $N$'); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_fair.pdf'); plt.close()
print(json.dumps({k: V[k] for k in ['beb20','fix20','ti20','ts20','jti20','jts20','b01i20','b01s20','b01i20pc','b01s20pc','beb20pc','fixcw','share','cw20','cw20_b001']}, default=float, indent=1))
print("val", [(n, round(S,3), round(bs,3)) for n,S,bs,P,bp in val]); print("fz", [(W, round(a,4), round(b,4), round(c,4)) for W,a,b,c in fz])
