# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Audit complete, real staged results; execution completion is not a QAM pass."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import time

ROOT=Path('/opt/divl-trqam')

def load(name):
    return json.loads((ROOT/'artifacts'/name).read_text(),parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonstandard JSON '+s)))

def main():
    required_reports=['phase0_beta_fp32_report.md','phase0b_beta_label_report.md','phase1_libero_training_entry.md',
        'expert_hdf5_audit.md','pilot_replay_report.md','critic_scalar_report.md','critic_divl_report.md',
        'critic_action_diagnostic.md','counterfactual_branch_report.md','current_stage_summary.md']
    checks={}
    for name in required_reports:
        assert (ROOT/'reports'/name).is_file() and (ROOT/'reports'/name).stat().st_size>0,name
    checks['required_reports']=required_reports
    for name in ['phase0_beta_fp32_metrics.json','beta_target_audit.json','beta_target_pilot_audit.json',
                 'four_arm_smoke_test.json','pilot_replay_audit.json','critic_pilot_training_complete.json']:
        assert load(name)['passed'],name
    checks['earlier_stage_gates']=True
    immutable=load('phase3_audited_manifest_hashes.json')
    for dataset,sha in immutable.items():
        assert hashlib.sha256((ROOT/'data'/dataset/'manifest.json').read_bytes()).hexdigest()==sha
    checks['frozen_manifests_unchanged']=True
    expert=load('expert_hdf5_manifest.json');assert expert['resolved']==0 and expert['unresolved']==500
    pilot=json.loads((ROOT/'data/pilot_libero_v1/manifest.json').read_text());assert pilot['episodes']==200
    checks['unresolved_experts_excluded']=True
    for steps in [250,500,1000,2000]:
        records=[]
        for method in ['scalar','divl']:
            folder=ROOT/'checkpoints/critic_pilot_v1'/method
            assert (folder/f'step{steps:04d}').is_dir()
            row=json.loads((folder/f'metrics_step{steps:04d}.json').read_text())
            assert row['batch_size']==256 and len(row['metrics'])==steps
            assert all(r['update_applied'] and all(math.isfinite(v) for v in r.values()) for r in row['metrics'])
            records.append(row)
        assert records[0]['sample_order_sha256']==records[1]['sample_order_sha256']
        configs=[{k:v for k,v in r['config'].items() if k!='value_type'} for r in records]
        assert configs[0]==configs[1] and not configs[0]['actor_updates']
    checks['matched_all_four_critic_budgets']=True
    benchmark=load('critic_batch_benchmark.json')
    assert len(benchmark)==8
    assert {(row['method'],row['batch']) for row in benchmark}=={(m,b) for m in ['scalar','divl'] for b in [256,512,1024,2048]}
    for row in benchmark:
        if row['status']=='completed':assert math.isfinite(row['samples_per_second']) and row['samples_per_second']>0
        else:assert row['status'] in ['oom','failed','timeout'] and row['samples_per_second'] is None
    checks['all_disposable_batch_profiles_retained']=True
    results=load('critic_action_metrics.json');groups=load('counterfactual_results.json')
    assert len(groups)==50
    assert len({(g['metadata']['task_id'],g['metadata']['initial_state_id']) for g in groups})==50
    assert all(sum(g['metadata']['task_id']==t for g in groups)==5 for t in range(10))
    total=0;initial_error=next_error=return_error=0.
    for group in groups:
        assert len(group['branches'])==20 and len(group['input_hashes'])==2
        for method in ['scalar','divl']:
            state_id=group['metadata']['state_id']
            path=ROOT/f'data/action_diagnostic_v1/{state_id}_{method}_step2000.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==group['input_hashes'][method]
        assert sum(r['kind']=='candidate' for r in group['branches'])==8
        assert sum(r['kind']=='gradient' for r in group['branches'])==12
        for row in group['branches']:
            assert math.isfinite(row['future_return']) and row['initial_state_error']<=1e-9 and row['proprio_error']<=1e-5
            initial_error=max(initial_error,row['initial_state_error'])
            if row['baseline_next_state_error'] is not None:
                next_error=max(next_error,row['baseline_next_state_error']);assert row['baseline_next_state_error']<=1e-7
            if 'baseline_future_error' in row:
                return_error=max(return_error,row['baseline_future_error']);assert row['baseline_future_error']<=1e-5
            total+=1
    assert total==1000 and (ROOT/'artifacts/counterfactual_results.parquet').stat().st_size>0
    checks['counterfactual_branch_counts_and_provenance']=True
    checks['snapshot_max_errors']={'initial':initial_error,'next_state':next_error,'baseline_future_return':return_error}
    complete=load('action_diagnostic_complete.json');assert complete['passed_execution'] and not complete['formal_actor_training_started']
    assert complete['decision']==results['decision']
    checks['formal_actor_training_absent']=True
    summary=(ROOT/'reports/current_stage_summary.md').read_text()
    assert all(f'| {i} |' in summary for i in range(1,19))
    checks['summary_answers_18_questions']=True
    manifest=load('code_change_manifest.json')
    for row in manifest['changes']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    checks['code_change_shas_match']=True
    checks['decision']=results['decision']
    output={'passed_execution_and_delivery_audit':True,'checks':checks,'utc_epoch':time.time(),
        'scope':'Complete artifacts and protocol checks; NOT a claim that either critic passes action evidence or DIVL beats Scalar.'}
    (ROOT/'artifacts/stage_deliverables_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
