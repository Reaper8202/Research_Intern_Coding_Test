"""D36 check: which event codes predict an anchor before 2015. Run from the repository root."""
import warnings, sys
from pathlib import Path
import lightgbm, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from ehrforecast.contracts import load_training_data
from ehrforecast.features import BagOfCodes
from ehrforecast.folds import fold_patients, load_folds
from ehrforecast.lightgbm_cv import FIXED
warnings.filterwarnings("ignore"); assert "torch" not in sys.modules
ds = Path("data/processed"); data = load_training_data(ds); ev, co, idx = data.inputs.events, data.inputs.cohort, data.labels.index
year = ev["time"].dt.year
g = ev.assign(y=year, k=ev["source"].astype(str)+":"+ev["code"].astype(str)).groupby("k")
tab = g.agg(rows=("y","size"), pts=("patient_id","nunique"), first=("y","min"), p05=("y", lambda s: s.quantile(.05)), last=("y","max"))
print("Common codes (>=100 patients) whose records start late; sorted by 5th-percentile year:")
print(tab[tab.pts>=100].sort_values("p05",ascending=False).head(14).to_string())
d = pd.read_parquet(ds/"code_descriptions.parquet"); print("description columns:", list(d.columns)[:5])
early = co.loc[idx,"anchor_date"].dt.year<2015; A = load_folds(ds); fit, held = fold_patients(A,0)
b = BagOfCodes().fit(ev,co.loc[fit]); xf,xh = b.transform(ev,co.loc[fit]), b.transform(ev,co.loc[held])
drop=("DALY","QALY","QOLS"); removed=[]
m0=lightgbm.LGBMClassifier(n_estimators=100,num_leaves=4,min_child_samples=50,importance_type="gain",**FIXED).fit(xf.to_numpy("float32"),early.loc[fit].astype(int))
imp0=pd.Series(m0.feature_importances_,index=xf.columns).sort_values(ascending=False); imp0=imp0/imp0.sum()
print(f"era AUROC fold0 with DALY/QALY/QOLS present: {roc_auc_score(early.loc[held],m0.predict_proba(xh.to_numpy('float32'))[:,1]):.3f} | top: {[(c,round(float(v),2)) for c,v in imp0.head(4).items()]}")
for round_ in range(6):
    keep=[c for c in xf.columns if not any(c.endswith(":"+k) for k in drop) and c not in removed]
    m=lightgbm.LGBMClassifier(n_estimators=100,num_leaves=4,min_child_samples=50,importance_type="gain",**FIXED).fit(xf[keep].to_numpy("float32"),early.loc[fit].astype(int))
    auc=roc_auc_score(early.loc[held],m.predict_proba(xh[keep].to_numpy("float32"))[:,1]); imp=pd.Series(m.feature_importances_,index=keep).sort_values(ascending=False); imp=imp/imp.sum()
    print(f"era AUROC fold0, removed {len(removed)} more: {auc:.3f} | top: {[(c,round(float(v),2)) for c,v in imp.head(4).items()]}")
    removed+=list(imp.head(4).index)
