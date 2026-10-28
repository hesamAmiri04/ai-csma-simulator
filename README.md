# csma_rl: clean slotted CSMA/CA simulator for CW-adaptation studies

Files: `csmasim.py` (simulator, policies, NumPy DQN, Bianchi check), `run_experiments.py` (density sweep).
Dependencies: numpy, matplotlib (scipy optional, for t-based CIs). No torch.

    python run_experiments.py --quick     # ~1 min smoke test, NOT for the paper
    python run_experiments.py             # full sweep: 10 seeds, N=2..20, saturated traffic
    python run_experiments.py --lam 0.01  # unsaturated Bernoulli arrivals
    python run_experiments.py --actions paper --reward paper   # reproduce the paper's DQN setup

Outputs `results/raw.csv`, `results/summary.csv` (mean, 95% CI), `results/density_sweep.png`.

## Model
Bianchi-style virtual slots: idle slot = 1, success or collision = `tx_slots` (default 20).
Counter-based backoff uniform in {0..CW-1}, counters freeze during busy periods.
Throughput = successful_tx * tx_slots / time. p_coll = collided attempts / attempts.

## Validation (saturated, tx_slots=20)
* BEB vs Bianchi: throughput within 1.7-3.5% (N=2,5,10,20), p_coll within 0.002-0.02.
* Fixed CW, freezing disabled (`freeze=False`): matches analytic tau=2/(W+1) within 0.3%
  -> counter mechanics are correct. With freezing, throughput is ~3% below the analytic value
  (Bianchi's decoupling approximation ignores freezing correlations).

## Protocols
beb, fixed_best (tuned on separate seeds), analytic (CW from tau*, knows N), dqn_indep,
dqn_shared (parameter sharing), dqn_one (one learner among BEB nodes).
Training in slots, then greedy evaluation (eps=0, no learning) on fresh seeds.

## Caveats
* DQN observation is 5 features: busy ratio, collision flag, queue, success ratio, last CW (log).
  The paper has 4 (no last CW).
* Rewards (`--reward`): `time` = +1 success / -1 collision - beta*elapsed/T (default beta=0.1 makes agents greedy:
  they pick CWs that are too small; beta=0.01 works much better), `paper` = +5/-10/-0.5*busy,
  `team` = system utilisation over the node's decision interval (needs global info only during training).
* Tune hyperparameters with `--seed-offset 100`, report final numbers with the default offset 0.
* The runner logs the mean chosen CW (`mean_cw`, `node0_cw`, `others_cw`) and plots it in the 4th panel.
* The paper's action set {4..64} cannot reach the analytic optimum for N >= ~10
  (analytic CW = 62 at N=10, 125 at N=20 for tx_slots=20). Default here is {8..1024}.
