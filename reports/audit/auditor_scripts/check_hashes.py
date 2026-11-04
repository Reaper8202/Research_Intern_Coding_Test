import hashlib, json, sys, os
from pathlib import Path
repo = Path(sys.argv[1])
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
m = json.load(open(repo / 'reports/final/manifest.json'))
bases = {'code_hashes': repo, 'comparison_hashes': repo, 'model_hashes': repo / 'models', 'dataset_hashes': repo / 'data/processed'}
total = ok = missing = diff = 0
for section, base in bases.items():
    s_ok = s_missing = s_diff = 0
    diffs = []
    for rel, h in m[section].items():
        p = base / rel
        if not p.exists():
            p2 = repo / rel
            p = p2 if p2.exists() else p
        total += 1
        if not p.exists():
            s_missing += 1; diffs.append(('MISSING', rel)); continue
        if sha(p) == h:
            s_ok += 1
        else:
            s_diff += 1; diffs.append(('DIFF', rel))
    print(f"{section}: {len(m[section])} listed, {s_ok} match, {s_diff} differ, {s_missing} missing")
    for d in diffs[:40]:
        print('   ', *d)
    ok += s_ok; missing += s_missing; diff += s_diff
print(f"TOTAL listed {total}: match {ok}, differ {diff}, missing {missing}")
# final config hash
print('final_config_sha256 matches configs/final.toml:', sha(repo/'configs/final.toml') == m['final_config_sha256'])
for rel, h in m['member_config_sha256'].items():
    print('member config', rel, 'matches:', sha(repo/rel) == h)
# package.json hashes vs models/hf files
pkg = json.load(open(repo / 'models/hf/package.json'))
p_ok = p_bad = 0
for rel, h in pkg['file_hashes'].items():
    if sha(repo / 'models/hf' / rel) == h: p_ok += 1
    else: p_bad += 1; print('PACKAGE DIFF', rel)
print(f"package.json: {len(pkg['file_hashes'])} listed, {p_ok} match, {p_bad} differ")
# cross-check: package member best.pt hashes equal manifest model_hashes
cross_ok = cross_bad = 0
for rel, h in pkg['file_hashes'].items():
    if rel.startswith('transformer/') and rel in m['model_hashes']:
        if m['model_hashes'][rel] == h: cross_ok += 1
        else: cross_bad += 1; print('CROSS DIFF', rel)
print(f"package transformer files also in manifest: {cross_ok} equal, {cross_bad} differ")
print('predictions.csv sha256:', sha(repo/'predictions.csv'))
print('package.json keys:', [k for k in pkg.keys()])
for k in pkg:
    if k not in ('file_hashes', 'members', 'targets'):
        print('  ', k, '=', json.dumps(pkg[k])[:300])
print('members[0]:', json.dumps(pkg['members'][0])[:400])
