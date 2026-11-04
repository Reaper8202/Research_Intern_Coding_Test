import json, sys
import pandas as pd
repo = sys.argv[1]
def J(p): return json.load(open(f'{repo}/{p}'))
def show(label, obj, depth=0):
    print(label, json.dumps(obj, indent=None)[:900] if not isinstance(obj, (int,float,str)) else obj)
vs = J('reports/val/val_summary.json')
print('== reports/val/val_summary.json keys:', list(vs.keys()))
for k, v in vs.items():
    if k in ('per_target', 'targets'): continue
    s = json.dumps(v)
    print(f'  {k}: {s[:1200]}')
print()
va = J('reports/val/attempts/168b78f5b5d8/val_summary.json')
print('== attempt-1 val_summary keys:', list(va.keys()))
for k in ('models', 'metrics', 'final', 'transformer', 'bundle', 'attempt'):
    if k in va: print(f'  {k}: {json.dumps(va[k])[:900]}')
print()
st = J('reports/stacking/clean/stacking.json')
print('== stacking.json keys:', list(st.keys()))
print(json.dumps(st)[:2500])
print()
ca = J('reports/calibration/stack/calibration.json')
print('== calibration.json:', json.dumps(ca)[:1500])
print()
print('== representation small-v2-base summary.csv:'); print(open(f'{repo}/reports/representation/small-v2-base/summary.csv').read()[:1500])
f0 = J('reports/representation/small-v2-base/fold0.json')
print('== fold0.json keys:', list(f0.keys()))
for k in f0:
    if 'refill' in k.lower() or 'merge' in k.lower() or 'vocab' in k.lower() or 'trunc' in k.lower() or 'retention' in k.lower() or 'size' in k.lower():
        print('  ', k, json.dumps(f0[k])[:300])
print()
cv1 = J('reports/transformer/small-v1/cv_summary.json')
print('== small-v1 cv_summary keys:', list(cv1.keys()))
print(json.dumps(cv1)[:2000])
print()
r3 = J('reports/transformer/round3/experiments.json')
print('== round3 experiments.json:', json.dumps(r3)[:1500])
print()
mo = J('reports/dataset/modalities.json'); print('== modalities.json:', json.dumps(mo)[:800])
dr = J('reports/dataset/dataset_report.json'); print('== dataset_report keys:', list(dr.keys())); print(json.dumps(dr)[:1500])
print()
cvc1 = J('reports/transformer/small-v2-c1/cv_summary.json'); print('== small-v2-c1 cv_summary:', json.dumps(cvc1)[:1500])
print()
lg = J('reports/lightgbm_clean/cv_summary.json'); print('== lightgbm_clean cv_summary:', json.dumps(lg)[:1200])
ec = J('reports/lightgbm_clean/ensemble_comparison.json'); print('== ensemble_comparison:', json.dumps(ec)[:1200])
lf = J('reports/lightgbm_final/lightgbm_final.json'); print('== lightgbm_final.json:', json.dumps(lf)[:800])
