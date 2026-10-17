# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Select five diverse restore checks from the fixed diagnostic state inventory."""
import argparse
import json
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env, LIBERO_ENV_RESOLUTION
from libero_snapshot_v2 import migrate_v1_for_branch, restore_snapshot

ROOT=Path('/opt/divl-trqam')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-inventory',action='store_true');args=parser.parse_args()
    fixed=json.loads((ROOT/'artifacts/static_action_scalar_step2000.json').read_text())['records']
    selected=[r for r in fixed if r['selected_for_branches']]
    failed=json.loads((ROOT/'artifacts/reproduction_failed_state.json').read_text())['state_id']
    results=[]
    cache_path=ROOT/'artifacts/reproduction_model_inventory_progress.json'
    cache={r['state_id']:r for r in json.loads(cache_path.read_text())['inventory']} if args.resume_inventory and cache_path.exists() else {}
    suite=benchmark.get_benchmark_dict()['libero_10']()
    for meta in selected:
        if meta['state_id'] in cache:
            row=cache[meta['state_id']]
            assert row['snapshot_version']==2
            proof=row['model_reconstruction_proof'];assert proof['matched_fingerprint']==proof['expected_fingerprint']
            results.append(row);continue
        print('INSPECT_GATE_STATE',meta['state_id'],flush=True)
        tid=meta['task_id']
        env,_=_get_libero_env(suite.get_task(tid),LIBERO_ENV_RESOLUTION,meta['seed'])
        try:
            env.reset()
            for meta in [meta]:
                with np.load(meta['episode_path'],allow_pickle=False) as f:
                    flat=f['snapshot_states'][meta['index']];metadata=f['snapshot_metadata'][meta['index']];xml=f['model_xml']
                migrated=migrate_v1_for_branch(env,flat,metadata,xml);restore_snapshot(env,flat,migrated)
                gripper=env.robots[0].gripper
                gripped=[]
                for name,obj in getattr(env.env,'objects_dict',{}).items():
                    try:
                        if env.env._check_grasp(gripper=gripper,object_geoms=obj.contact_geoms):gripped.append(name)
                    except (AttributeError,TypeError):pass
                contacts=[];robot_contacts=[]
                for c in env.sim.data.contact[:env.sim.data.ncon]:
                    a=env.sim.model.geom_id2name(c.geom1);b=env.sim.model.geom_id2name(c.geom2)
                    row=[a,b,float(c.dist)];contacts.append(row)
                    if ('gripper' in str(a))!=('gripper' in str(b)):robot_contacts.append(row)
                primary_before=np.asarray(flat)
                row={**meta,'grasped_objects':gripped,'ncon':int(env.sim.data.ncon),'contacts':contacts,
                     'gripper_contacts':robot_contacts,'snapshot_version':2,
                     'model_reconstruction_proof':json.loads(migrated)['migration']['model_reconstruction_proof']}
                results.append(row)
                (ROOT/'artifacts/reproduction_model_inventory_progress.json').write_text(json.dumps({'states':len(results),'target':50,'inventory':results},indent=2))
        finally:env.close()
    assert len(results)==50
    for row in results:
        with np.load(row['episode_path'],allow_pickle=False) as f:
            row['steps_to_recorded_termination']=int(f['lengths'][row['index']:].sum())
    chosen=[next(r for r in results if r['state_id']==failed)]
    chosen[0]['gate_role']='previously_failed'
    predicates=[('free_space',lambda r:not r['grasped_objects'] and not r['gripper_contacts']),
                ('object_contact',lambda r:bool(r['gripper_contacts'])),
                ('grasp',lambda r:bool(r['grasped_objects'])),
                ('near_terminal',lambda r:r['steps_to_recorded_termination']<=50)]
    for role,pred in predicates:
        pool=[r for r in results if pred(r) and r['state_id'] not in [x['state_id'] for x in chosen]]
        if not pool:raise AssertionError('No qualifying gate state for '+role)
        if role=='near_terminal':pool.sort(key=lambda r:(r['steps_to_recorded_termination'],r['state_id']))
        row=pool[0];row['gate_role']=role;chosen.append(row)
    assert len(chosen)==5
    result={'states':chosen,'selection_scope':'Fixed50 dev diagnostic inventory; contact/grasp/source time to horizon only, no return outcome or critic-score selection',
            'all_state_inventory':results,'repetitions_per_state':20,'future_return_tolerance':1e-5}
    (ROOT/'artifacts/reproduction_gate_selection.json').write_text(json.dumps(result,indent=2))
    print('GATE_SELECTION',json.dumps([{k:r[k] for k in ['state_id','gate_role','grasped_objects','ncon','remaining_steps']} for r in chosen]),flush=True)

if __name__=='__main__':main()
