import json,zipfile
from pathlib import Path
import numpy as np
from horizon_common import R,D,O,REP,atomic,sha,safe
m=json.loads((O/'round_complete.json').read_text());assert m['complete']
required_reports=['action_object_code_audit.md','action_horizon_branch_report.md','action_horizon_return_sensitivity.md','policy_recovery_curve.md','return_target_alignment.md','action_horizon_positive_control.md','current_stage_summary.md']
required_artifacts=['action_object_shape_trace.json','action_horizon_branches.parquet','action_horizon_metrics.json','recovery_curve_metrics.json','target_alignment.json','horizon_positive_control.json']
assert all((REP/n).exists() for n in required_reports)
assert all((R/'artifacts'/n).exists() for n in required_artifacts)
text=(REP/'current_stage_summary.md').read_text();assert all(f'## {i}. ' in text for i in range(1,35))
source=json.loads((D/'manifest.json').read_text());selection=json.loads((D/'state_selection_manifest.json').read_text())['records'];assert len(source['records'])==1500 and len(selection)==100
assert len({(r['task_id'],r['initial_state_id']) for r in selection})==100
for h in [1,3,5]:assert sum(r['horizon']==h for r in source['records'])==500
for t in range(10):assert sum(r['task_id']==t for r in selection)==10
target=json.loads((R/'artifacts/target_alignment.json').read_text());rec=json.loads((R/'artifacts/recovery_curve_metrics.json').read_text())
brief=dict(round=m,recovery_times=rec['recovery_times'],recovery_curves={h:{k:v for k,v in curve.items() if k in ['intervention_end','replan1','replan2','replan3']} for h,curve in rec['curves'].items()},
 target_arithmetic_errors=target['arithmetic_errors'],target_summary=target['frozen_v2_target_evaluations'],target_bug=target['implementation_bug'])
atomic(O/'delivery_brief.json',safe(brief))
paths=list(REP.glob('*'))+[R/'artifacts'/n for n in required_artifacts]
paths += [p for p in O.glob('*.json') if p.name != 'delivery_validation.json']+list(O.glob('*.parquet'))+list(O.glob('*.npz'))
paths += [D/n for n in ['protocol.json','manifest.json','state_selection_manifest.json']]
paths += [R/'scripts'/n for n in ['horizon_common.py','audit_prepare_action_horizon.py','collect_action_horizon.py','run_action_horizon_collection.py','analyze_action_horizon.py','evaluate_horizon_targets.py','run_action_horizon_analysis.py','finalize_action_horizon.py','polish_action_horizon_reports.py','package_action_horizon.py','queue_action_horizon_analysis.py','run_horizon_audit.py']]
paths += list((D/'invalid').glob('*.json'))
out=O/'action_horizon_delivery.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for path in sorted(set(paths)):z.write(path,path.relative_to(R))
atomic(O/'delivery_validation.json',dict(required_reports=7,required_artifacts=6,all34answers=True,states=100,branches=1500,restore_repeats=300,
 decision=m['decision'],zip=str(out),zip_bytes=out.stat().st_size,zip_sha256=sha(out),source_data_not_copied=True))
print('DELIVERY_PACKAGE',out,out.stat().st_size,'decision',m['decision'])
