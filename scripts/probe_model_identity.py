# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Locate task post-processing model state omitted by fresh reset."""
import argparse
import difflib
import json
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION,LIBERO_DUMMY_ACTION
from libero_snapshot import model_fingerprint

ROOT=Path('/opt/divl-trqam')
FIELDS=['body_mass','body_inertia','body_pos','body_quat','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','dof_armature','dof_damping','jnt_range','actuator_gainprm','actuator_biasprm']

def main():
    p=argparse.ArgumentParser();p.add_argument('--state-id',default='task03_init009_rep0_chunk017');args=p.parse_args()
    meta=next(v for v in json.loads((ROOT/'artifacts/static_action_scalar_step2000.json').read_text())['records'] if v['state_id']==args.state_id)
    with np.load(meta['episode_path'],allow_pickle=False) as f:e={k:f[k] for k in ['snapshot_states','snapshot_metadata','model_xml','actions','lengths']}
    target=json.loads(str(e['snapshot_metadata'][meta['index']]))['model_fingerprint']
    suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(meta['task_id'])
    env,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,meta['seed']);results=[]
    def record(name):
        result={'stage':name,'fingerprint':model_fingerprint(env),'match':model_fingerprint(env)==target}
        results.append(result);print('MODEL_STATE',json.dumps(result),flush=True)
        return {k:np.asarray(getattr(env.sim.model,k)).copy() for k in FIELDS}
    try:
        env.reset();initial=record('fresh_reset')
        source_rng=json.loads(str(e['snapshot_metadata'][meta['index']]))['numpy_rng']
        current_rng=np.random.get_state()
        print('RNG_CURSOR',json.dumps({'recorded_index':source_rng[2],'fresh_index':int(current_rng[2]),
            'mt_array_equal':bool(np.array_equal(current_rng[1],np.asarray(source_rng[1],np.uint32))),
            'source_first_index':json.loads(str(e['snapshot_metadata'][0]))['numpy_rng'][2]}),flush=True)
        env.set_init_state(suite.get_task_init_states(meta['task_id'])[meta['initial_state_id']]);second=record('official_initial_state_postprocess')
        for _ in range(10):env.step(LIBERO_DUMMY_ACTION)
        third=record('ten_recorded_settling_steps')
        for c in range(meta['index']):
            for a in e['actions'][c][:int(e['lengths'][c])]:env.step(a.tolist())
        fourth=record('recorded_action_prefix')
        changes={}
        for label,data in [('after_init',second),('after_settling',third),('after_prefix',fourth)]:
            changes[label]={k:{'max_error':float(np.max(np.abs(data[k]-initial[k]),initial=0)),
                               'changed_indices':np.argwhere(data[k]!=initial[k]).tolist()} for k in FIELDS if not np.array_equal(data[k],initial[k])}
        current=env.sim.model.get_xml()
        diff='\n'.join(difflib.unified_diff(str(e['model_xml']).splitlines(),current.splitlines(),fromfile='recorded_episode_model',tofile='reconstructed_model',n=1))
        directory=ROOT/'data/reproduction_debug_v1'/meta['state_id'];directory.mkdir(parents=True,exist_ok=True)
        (directory/'model_xml_diff.txt').write_text(diff)
        result={'state_id':meta['state_id'],'expected_fingerprint':target,'stages':results,'changes':changes,'model_xml_diff_path':str(directory/'model_xml_diff.txt')}
        if not any(v['match'] for v in results):
            # Fixture placement is omitted by MuJoCo qpos/qvel. The full saved
            # MT19937 block and cursor permit replay of preceding placement RNG
            # draws within that block; accept ONLY exact physical fingerprint.
            matched=[]
            for cursor in range(0,min(source_rng[2]+2,624),2):
                np.random.set_state((source_rng[0],np.asarray(source_rng[1],np.uint32),cursor,0,0.))
                env.reset()
                if model_fingerprint(env)==target:
                    matched.append({'cursor':cursor,'post_index':int(np.random.get_state()[2])})
                    print('EXACT_MODEL_RECONSTRUCTION',json.dumps(matched[-1]),flush=True)
                    break
            result['rng_reconstruction_matches']=matched
        (ROOT/'artifacts/model_state_postprocess_audit.json').write_text(json.dumps(result,indent=2))
        print('MODEL_CHANGES',json.dumps(changes),flush=True)
    finally:env.close()

if __name__=='__main__':main()
