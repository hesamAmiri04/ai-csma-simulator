import json
from string import Template
V = json.load(open('V.json'))
NS = [2, 4, 6, 8, 10, 12, 16, 20]
class T(Template): delimiter = '@@'

pi = {n: V['pair'][f'team_dqn_indep_{n}'] for n in NS}; ps = {n: V['pair'][f'team_dqn_shared_{n}'] for n in NS}
fi = {n: V['pairfix'][f'team_dqn_indep_{n}'] for n in NS}; fs = {n: V['pairfix'][f'team_dqn_shared_{n}'] for n in NS}
b1 = {n: V['pair'][f'b001_dqn_indep_{n}'] for n in NS}; b1s = {n: V['pair'][f'b001_dqn_shared_{n}'] for n in NS}
L = lambda xs: ", ".join(str(x) for x in xs)
sig_i = [n for n in NS if pi[n][1] < 0.01 and pi[n][0] > 0]
ns_i = [n for n in NS if not (pi[n][1] < 0.05 and pi[n][0] > 0)]
sig_s = [n for n in NS if ps[n][1] < 0.05 and ps[n][0] > 0]
tie_i = [n for n in NS if fi[n][1] >= 0.05]
above_fix_i = [n for n in NS if fi[n][1] < 0.05 and fi[n][0] > 0]
worse_b1 = [n for n in NS if b1[n][1] < 0.05 and b1[n][0] < 0]
better_b1 = [n for n in NS if b1[n][1] < 0.05 and b1[n][0] > 0]
ONE_DEV='0.01'
d = dict(
 beb2=f"{V['beb2']:.3f}", beb10=f"{V['beb10']:.3f}", beb20=f"{V['beb20']:.3f}",
 fix20=f"{V['fix20']:.3f}", ti20=f"{V['ti20']:.3f}", ts20=f"{V['ts20']:.3f}", ti10=f"{V['ti10']:.3f}", ts10=f"{V['ts10']:.3f}",
 jti10=f"{V['jti10']:.2f}", jti20=f"{V['jti20']:.2f}", jts10=f"{V['jts10']:.2f}", jts20=f"{V['jts20']:.2f}",
 b01i20=f"{V['b01i20']:.2f}", b01s20=f"{V['b01s20']:.2f}", b01i20pc=f"{V['b01i20pc']:.2f}", b01s20pc=f"{V['b01s20pc']:.2f}", beb20pc=f"{V['beb20pc']:.2f}",
 b001i2=f"{V['b001i2']:.3f}", b001i20=f"{V['b001i20']:.3f}",
 dmin=f"{min(pi[n][0] for n in NS):.3f}", dmax=f"{max(pi[n][0] for n in NS):.3f}", d20=f"{pi[20][0]:.3f}", p20=f"{pi[20][1]:.3f}",
 ds20=f"{ps[20][0]:.3f}", ps20=f"{ps[20][1]:.3f}",
 sig_i=L(sig_i), ns_i=L(ns_i), sig_s=L(sig_s), above_fix_i=L(above_fix_i), worse_b1=L(worse_b1), better_b1=L(better_b1),
 tie_i=L(tie_i), one_dev=ONE_DEV, fixcw2=V['fixcw']['2'], fixcw10=V['fixcw']['10'], fixcw20=V['fixcw']['20'],
 sh10_01=f"{V['share']['b01']['10']:.1f}", sh20_01=f"{V['share']['b01']['20']:.1f}",
 sh20_001=f"{V['share']['b001']['20']:.1f}", sh20_t=f"{V['share']['team']['20']:.2f}", sh10_t=f"{V['share']['team']['10']:.2f}",
 cw20_i=f"{V['cw20']['dqn_indep']:.0f}", cw20_s=f"{V['cw20']['dqn_shared']:.0f}", cw20_a=f"{V['cw20']['analytic']:.0f}", cw20_b=f"{V['cw20']['beb']:.0f}",
 cw20_i001=f"{V['cw20_b001']['dqn_indep']:.0f}",
 verr=f"{max(abs(100*(S-b)/b) for n,S,b,P,bp in V['val']):.1f}",
 nf=f"{max(abs(100*(c-a)/a) for W,a,c,w in V['fz']):.1f}", wf=f"{max(abs(100*(w-a)/a) for W,a,c,w in V['fz']):.1f}",
)
import re
tex = re.sub(r'@@(\w+)@@', r'@@{\1}', open('template.tex').read())
open('/mnt/user-data/outputs/paper/main.tex', 'w').write(T(tex).substitute(d))
print(d)
