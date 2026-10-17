"""Runtime parallelism only; reuse the frozen collector/data protocol verbatim."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import argparse,json,time,hashlib
from collect_action_sensitive_replay_v2 import episode_group,R,D

def main():
    p=argparse.ArgumentParser();p.add_argument('--ports',default='8027,8028,8029');p.add_argument('--workers',type=int,default=9);args=p.parse_args()
    ports=[int(v) for v in args.ports.split(',')];source=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text());protocol=json.loads((D/'protocol.json').read_text())
    assert protocol['collector_sha256']==hashlib.sha256((R/'scripts/collect_action_sensitive_replay_v2.py').read_bytes()).hexdigest()
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text());assert protocol['split_hash']==split['sha256']
    collected=[];start=time.monotonic();completed=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(episode_group,row,[.1,.3,.5,.7,.9],ports[n%len(ports)],str(R/'artifacts/replay_action_audit_v3/public_features.npz')) for n,row in enumerate(source['records'])]
        for job in as_completed(jobs):
            collected.extend(job.result());completed+=1
            elapsed=time.monotonic()-start
            status=dict(status='collecting',outcomes=len(collected),states=len(collected)//5,target_outcomes=5000,target_states=1000,completed_source_episodes=completed,total_source_episodes=200,elapsed_seconds=elapsed,outcomes_per_minute=len(collected)/elapsed*60,workers=args.workers,ports=ports)
            (D/'progress.json').write_text(json.dumps(status));print('PROGRESS',json.dumps(status),flush=True)
    assert len(collected)==5000 and len({v['episode_id'] for v in collected})==5000 and len({v['state_id'] for v in collected})==1000
    inventory=dict(version='action_sensitive_replay_v2',split_hash=source['split_hash'],pilot=False,records=sorted(collected,key=lambda x:x['episode_id']),states=1000,outcomes=5000,elapsed_seconds=time.monotonic()-start,
        purpose='critic train-only local intervention supervision; no actor updates',all_outcomes_retained=True,collector_sha256=protocol['collector_sha256'],runtime=dict(workers=args.workers,ports=ports),expert_included=0,dev_states_used=False)
    (D/'manifest.json').write_text(json.dumps(inventory,indent=2));(D/'progress.json').write_text(json.dumps(dict(status='complete',states=1000,outcomes=5000,elapsed_seconds=inventory['elapsed_seconds'])))
    print('COLLECTION_COMPLETE',json.dumps({k:v for k,v in inventory.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
