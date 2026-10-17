# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Immutable official init-state split; expert provenance must be resolved explicitly."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
from libero.libero import benchmark

ROOT=Path('/opt/divl-trqam')
DATA=Path('/home/researcher/lwd_libero_long/data/demos/libero_10')


def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    assert json.loads((ROOT/'artifacts/four_arm_smoke_test.json').read_text())['passed']
    suite=benchmark.get_benchmark_dict()['libero_10']()
    path=ROOT/'configs/splits/libero_factorial_v1.json'
    states={};tasks={}
    for i in range(suite.n_tasks):
        task=suite.get_task(i);initial=np.asarray(suite.get_task_init_states(i));states[task.name]=initial
        n=len(initial)
        if n<5:raise ValueError('Insufficient official initial states to make three splits')
        order=np.random.default_rng(20261007+i).permutation(n).tolist()
        ntrain=int(n*.6);ndev=int(n*.2)
        tasks[task.name]={'task_id':i,'official_initial_state_count':n,'initial_state_array_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),
            'train':sorted(order[:ntrain]),'dev':sorted(order[ntrain:ntrain+ndev]),'eval':sorted(order[ntrain+ndev:])}
        a,b,c=[set(tasks[task.name][k]) for k in ['train','dev','eval']]
        assert not a&b and not a&c and not b&c and a|b|c==set(range(n))
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT/'third_party/libero',capture_output=True,text=True,check=True).stdout.strip()
    if path.exists():
        split=json.loads(path.read_text());expected=split['sha256'];data={k:v for k,v in split.items() if k!='sha256'}
        assert canonical_hash(data)==expected
        assert split['tasks']==tasks, 'Existing frozen split conflicts; never overwrite it'
    else:
        split={'schema':'libero_factorial_v1','suite':'libero_10','seed':20261007,'created_at_utc':datetime.now(timezone.utc).isoformat(),
               'libero_commit':commit,'task_order':[suite.get_task(i).name for i in range(suite.n_tasks)],'tasks':tasks,
               'pretraining_limitation':'Split isolates new replay/adaptation and evaluation; official pi05_libero pretraining provenance is not claimed disjoint.'}
        split['sha256']=canonical_hash(split);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(split,indent=2)+'\n')
    records=[];hashes=set();duplicates=[]
    for task_name,initial in states.items():
        file=DATA/(task_name+'_demo.hdf5')
        if not file.exists():raise FileNotFoundError(file)
        with h5py.File(file,'r') as h:
            for key in sorted(h['data'],key=lambda k:int(k.split('_')[-1])):
                e=h['data'][key];n=len(e['actions']);action=np.asarray(e['actions']);state=np.asarray(e['states'])
                digest=hashlib.sha256(action.tobytes()+state.tobytes()).hexdigest()
                if digest in hashes:duplicates.append(task_name+'/'+key)
                hashes.add(digest)
                obs=e['obs'];obs_inventory={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in obs.items()}
                rgb=[k for k in obs if 'rgb' in k or 'image' in k]
                assert action.shape==(n,7) and np.isfinite(action).all()
                assert len(rgb)>=2 and all(len(obs[k])==n for k in rgb)
                # Decode beginning/middle/end to validate actual RGB chunks.
                for k in rgb:
                    for index in sorted(set([0,n//2,n-1])):
                        image=np.asarray(obs[k][index]);assert image.dtype==np.uint8 and image.shape[-1]==3
                proprio=[k for k in obs if any(s in k for s in ['ee','gripper','joint'])]
                assert proprio and all(len(obs[k])==n for k in proprio)
                matches=[];nearest_error=None;nearest_id=None
                if state.shape[-1]==initial.shape[-1]:
                    errors=np.max(np.abs(initial-state[0]),axis=-1);nearest_id=int(errors.argmin());nearest_error=float(errors[nearest_id])
                    matches=np.flatnonzero(errors<1e-6).tolist()
                mapped=matches[0] if len(matches)==1 else None
                split_name=next((s for s in ['train','dev','eval'] if mapped in tasks[task_name][s]),'unresolved')
                reward=np.asarray(e['rewards']) if 'rewards' in e else None
                done=np.asarray(e['dones']) if 'dones' in e else None
                records.append({'task_name':task_name,'episode_id':key,'path':str(file),'trajectory_hash':digest,'length':n,
                    'action_dimension':7,'rgb_fields':rgb,'proprio_fields':proprio,'observation_inventory':obs_inventory,
                    'initial_state_candidates':matches,'mapped_initial_state_id':mapped,'nearest_initial_state_id':nearest_id,
                    'nearest_initial_state_max_error':nearest_error,'split':split_name,
                    'success':bool(reward[-1]>0) if reward is not None else None,'success_evidence':'last recorded reward; no implicit all-success label',
                    'last_done':bool(done[-1]) if done is not None else None,
                    'eligible_for_train':split_name=='train','included_in_pilot':False})
    assert not duplicates,duplicates
    result={'episodes':len(records),'per_task':{t:sum(e['task_name']==t for e in records) for t in tasks},'duplicate_count':len(duplicates),
        'resolved':sum(e['mapped_initial_state_id'] is not None for e in records),'unresolved':sum(e['mapped_initial_state_id'] is None for e in records),
        'eligible_train':sum(e['eligible_for_train'] for e in records),'split_hash':split['sha256'],
        'provenance_method':'exact same task flattened initial simulator state match, max error <1e-6; ambiguous matches unresolved',
        'pilot_expert_inclusion':False,'episodes_manifest':records}
    (ROOT/'artifacts/expert_hdf5_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/expert_hdf5_audit.md').write_text('# Expert HDF5 and immutable split audit\n\n'
        f'{len(records)} trajectories; duplicates {len(duplicates)}; resolved initial-state provenance {result["resolved"]}; unresolved {result["unresolved"]}.\n\n'
        'Task-specific train/dev/eval initial IDs are disjoint and frozen in configs/splits/libero_factorial_v1.json.\n\n'
        'Unresolved expert pool is excluded from training. First pilot uses only deployment rollouts, even if some experts resolve.\n\n'
        'RGB shapes and first/middle/last frames decoded; full RGB chunk-by-chunk integrity is not claimed.\n')
    (ROOT/'reports/current_stage_summary.md').write_text('# Current stage\n\nPhases 0/0B/1 passed. Phase 2 split and expert audit completed.\n\n'
        f'Expert provenance: resolved {result["resolved"]}, unresolved {result["unresolved"]}. Pilot excludes experts.\n\n'
        'Next: 200 train deployment episodes and 100 dev frozen characterization episodes. Final eval IDs remain unused.\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes_manifest'},indent=2),flush=True)


if __name__=='__main__':main()
