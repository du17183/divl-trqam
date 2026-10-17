# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Run the required static update250 diagnostic early; no extra training budget."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
from run_action_diagnostic_after_critics import ensure_policy_server,static_diagnostic

ROOT=Path('/opt/divl-trqam')

def main():
    paths=[ROOT/f'checkpoints/critic_pilot_v1/{m}/metrics_step0250.json' for m in ['scalar','divl']]
    for _ in range(480):
        if all(p.exists() for p in paths):
            try:records=[json.loads(p.read_text()) for p in paths]
            except json.JSONDecodeError:time.sleep(1);continue
            break
        time.sleep(15)
    else:raise TimeoutError('No matched update250 checkpoint')
    assert records[0]['sample_order_sha256']==records[1]['sample_order_sha256']
    assert all(r['batch_size']==256 and len(r['metrics'])==250 for r in records)
    ensure_policy_server()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(static_diagnostic,m,250,gpu) for m,gpu in [('scalar',3),('divl',4)]]
        for job in jobs:job.result()
    (ROOT/'artifacts/first_checkpoint_static_probe.json').write_text(json.dumps({'passed_execution':True,'steps':250,
        'states_per_method':100,'candidate_actions':8,'physical_gpus':[3,4],
        'scope':'The same required static diagnostic is executed early and SHA-verified for reuse. No actor updates, no true-ranking conclusion, no additional training updates.'}))
    print('FIRST_STATIC_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
