# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Five independent states x20 full original-action continuations, fail closed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from debug_branch_reproduction import trace

ROOT=Path('/opt/divl-trqam')

def compact(run):
    out={k:v for k,v in run.items() if k!='checks'}
    out['max_anchor_state_error']=max((r.get('state_error',0) for r in run['checks']),default=0)
    out['max_proprio_error']=max((r.get('proprio_error',0) for r in run['checks']),default=0)
    out['max_pixel_error']=max((r.get('pixel_max',0) for r in run['checks']),default=0)
    out['max_plan_error']=max((r.get('plan_error',0) for r in run['checks']),default=0)
    out['max_executed_action_error']=max((r.get('executed_action_error',0) for r in run['checks']),default=0)
    out['max_reward_error']=max((r.get('reward_error',0) for r in run['checks']),default=0)
    return out

def case(meta,port):
    client=WebsocketClientPolicy('127.0.0.1',port);runs=[]
    destination=ROOT/'artifacts/reproduction_gate_cases';destination.mkdir(parents=True,exist_ok=True)
    output=destination/(meta['state_id']+'.json')
    for repeat in range(20):
        run=trace(meta,client,run_id=f'gate_v2_run{repeat:02d}',save=True,restore_version=2)
        runs.append(compact(run))
        output.write_text(json.dumps({'state_id':meta['state_id'],'role':meta['gate_role'],'runs':runs,'complete':len(runs)==20},indent=2))
        if not run['passed_return'] or run['first_divergence'] is not None:
            raise AssertionError(('Reproduction gate failed',meta['state_id'],repeat,run['future_return_error']))
    # Compare actual per-step measured traces with repeat0, independently of
    # unavailable original per-step physics. Original chunk anchors are above.
    repeat_errors={}
    with np.load(runs[0]['trace_path'],allow_pickle=False) as baseline:
        for run in runs[1:]:
            with np.load(run['trace_path'],allow_pickle=False) as other:
                for field in baseline.files:
                    a,b=baseline[field],other[field]
                    if a.shape!=b.shape:raise AssertionError('Repeat trace length mismatch '+field)
                    if a.dtype.kind in 'biufc':
                        error=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)),initial=0))
                    else:error=float(not np.array_equal(a,b))
                    repeat_errors[field]=max(repeat_errors.get(field,0),error)
    result={'state_id':meta['state_id'],'role':meta['gate_role'],'runs':runs,'complete':True,'repeats':20,
            'repeat_trace_max_errors':repeat_errors,'passed':all(v==0 for v in repeat_errors.values())}
    output.write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--port',type=int,default=8027);args=p.parse_args()
    assert json.loads((ROOT/'artifacts/counterfactual_pre_fix_manifest.json').read_text())['backup_complete']
    selection=json.loads((ROOT/'artifacts/reproduction_gate_selection.json').read_text())['states']
    assert len(selection)==5
    collected=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(case,meta,args.port) for meta in selection]
        for future in as_completed(futures):
            collected.append(future.result())
            (ROOT/'artifacts/reproduction_gate_progress.json').write_text(json.dumps({'completed_states':len(collected),'target':5,'results':collected,'elapsed_seconds':time.monotonic()-start},indent=2))
    names=['future_return_error','max_anchor_state_error','max_proprio_error','max_pixel_error','max_plan_error','max_executed_action_error','max_reward_error']
    maxima={k:max(v[k] for group in collected for v in group['runs']) for k in names}
    result={'passed':all(g['passed'] for g in collected) and maxima['future_return_error']<=1e-5,
            'snapshot_version':2,'states':5,'repeats_per_state':20,'total_repeats':100,'maximum_errors':maxima,
            'cases':collected,'old35_reuse':False,'rerun_all_fixed50_required':True,
            'root_cause':'(1) Extra mj_forward mutates original free-joint quaternion components by one ULP; contact amplifies into observation/action/success-time changes. Reapply exact original primary state after derived-state forward, preserve saved ctrl/warmstart/cache/controller. (2) LIBERO fixture placement is stored in mutable model.body_pos/body_quat, outside qpos/qvel. Seed-only fresh reset did not reconstruct task03_init009_rep0_chunk017 cabinet/wine-rack poses. Explicit V1 migration replays the recorded full MT19937 block, accepts only an identical full physical model fingerprint, and persists exact fixture arrays in V2.',
            'policy_fixed_input20':json.loads((ROOT/'artifacts/fixed_input_policy_reproduction.json').read_text()),
            'historical_state_limit':'Original only saved per-step actions/rewards and per-chunk simulator/controller/RGB. Per-step restored repetitions saved and compared; absent historical per-step states are explicitly unavailable.',
            'elapsed_seconds':time.monotonic()-start}
    (ROOT/'artifacts/reproduction_audit.json').write_text(json.dumps(result,indent=2))
    (ROOT/'reports/reproduction_gate_report.md').write_text('# Strict reproduction gate\n\n'+json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2)+'\n')
    (ROOT/'reports/reproduction_debug_report.md').write_text('# Branch reproduction debug\n\n'+result['root_cause']+'\n\n'
        'Old failed state: task01_init018_rep0_chunk009. Original protocol20/20 errors5.889675e-4. Frozen identical observation/noise20 calls action difference0. Primary reapplication ablation75 recorded steps simulator/RGB errors0.\n\n'
        'Version2 is explicit; old unversioned source is version1 and requires migration with na=0 activation proof. Original states/data remain immutable. Entire50-state branch dataset must be rerun; old35 remain provisional history.\n\n'
        'A second root cause was omitted mutable fixture model poses. Exact whole-reset RNG replay reproduces task03_init009_rep0_chunk017 at cursor0; for task03_init015_rep0_chunk026 independent fixture sampler replay recovers cabinet/rack poses at cursors40/46 and validates the complete original physical hash. Rounded XML coordinates only filter candidate RNG draws by formatting intervals; final poses are exact RNG values, never rounded XML. All50 fixture fingerprints are checked.\n\n'
        +json.dumps(result['maximum_errors'],indent=2)+'\n\n'+result['historical_state_limit']+'\n')
    if not result['passed']:raise AssertionError('Strict gate did not pass')
    print('REPRODUCTION_GATE_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
