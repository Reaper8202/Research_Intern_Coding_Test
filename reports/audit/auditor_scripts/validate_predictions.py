import sys, csv
import pandas as pd, numpy as np
repo = sys.argv[1]
raw = csv.reader(open(f'{repo}/predictions.csv', newline=''))
header = next(raw)
print('first column name:', repr(header[0]), '| n columns:', len(header))
pred = pd.read_csv(f'{repo}/predictions.csv', dtype={'patient_id': str})
splits = pd.read_csv(f'{repo}/data/raw/patient_splits.csv', dtype=str)
print('splits columns:', list(splits.columns), '| counts:', splits.iloc[:,1].value_counts().to_dict())
test_ids = set(splits[splits.iloc[:,1] == 'test'].iloc[:,0])
targets = pd.read_csv(f'{repo}/data/raw/target_conditions.csv', dtype=str)
print('targets columns:', list(targets.columns), '| n targets:', len(targets))
codes = list(targets.iloc[:,0].astype(str))
pred_codes = [c for c in pred.columns if c != 'patient_id']
print('rows:', len(pred), '| test ids in split:', len(test_ids))
print('id set equals test split:', set(pred['patient_id']) == test_ids)
print('duplicate ids:', pred['patient_id'].duplicated().sum())
print('columns equal target codes (as set):', set(pred_codes) == set(codes), '| same order:', pred_codes == codes)
print('missing codes:', sorted(set(codes) - set(pred_codes)), '| extra:', sorted(set(pred_codes) - set(codes)))
vals = pred[pred_codes]
print('all numeric:', all(np.issubdtype(d, np.number) for d in vals.dtypes))
print('NaN cells:', int(vals.isna().sum().sum()), '| min:', float(vals.min().min()), '| max:', float(vals.max().max()))
print('in [0,1]:', bool(((vals >= 0) & (vals <= 1)).all().all()))
print('exact zeros:', int((vals == 0).sum().sum()), '| exact ones:', int((vals == 1).sum().sum()))
print('rows with any blank raw cell:', sum(1 for r in csv.reader(open(f'{repo}/predictions.csv')) if any(c == '' for c in r)))
# test_anchors coverage
anchors = pd.read_csv(f'{repo}/data/raw/test_anchors.csv', dtype=str)
print('test_anchors ids == test split:', set(anchors.iloc[:,0]) == test_ids, '| n anchors:', len(anchors))
