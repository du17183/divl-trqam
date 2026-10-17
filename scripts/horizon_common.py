# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Bounded train-only action horizon diagnostic; never updates model weights."""
from pathlib import Path
import hashlib, json, numpy as np
R=Path('/opt/divl-trqam')
D=R/'data/action_horizon_alignment_20261008'
O=R/'artifacts/action_horizon_alignment_20261008'
REP=R/'reports/action_horizon_alignment_20261008'
SEED=2026100807
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def save_npz(p,**arrays):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
 with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
 tmp.replace(p)
def load(p,fields=None):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in (fields or f.files)}
def safe(x):
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.bool_):return bool(x)
 return x
def report(name,title,text):
 REP.mkdir(parents=True,exist_ok=True);(REP/name).write_text('# '+title+'\n\n'+text+'\n')
 (R/'reports'/name).write_text('# '+title+'\n\n'+text+'\n')
