import json
from horizon_common import R,D,O
for name in ['collection_status.json','analysis_status.json','target_scalar_progress.json','target_divl_progress.json','round_complete.json']:
 p=O/name
 if p.exists():print(name,p.read_text())
p=R/'artifacts/action_horizon_metrics.json'
if p.exists():
 x=json.loads(p.read_text());print('sensitivity',json.dumps({h:{k:v for k,v in s.items() if k!='state_metrics'} for h,s in x['horizons'].items()}));print('paired sensitivity',json.dumps(x['paired']))
p=R/'artifacts/horizon_positive_control.json'
if p.exists():
 x=json.loads(p.read_text());print('probes',json.dumps({h:{b:{k:v for k,v in m.items() if k!='states'} for b,m in s['metrics'].items()} for h,s in x['horizons'].items()}));print('paired probes',json.dumps(x['paired']))
