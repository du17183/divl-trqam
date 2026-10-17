# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Ablate primary-state rounding on original actions, without policy updates."""
import inspect
import json
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env, LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from libero_snapshot import restore_snapshot

ROOT=Path('/opt/divl-trqam')

def main():
    meta=json.loads((ROOT/'artifacts/reproduction_failed_state.json').read_text())
    with np.load(meta['episode_path'],allow_pickle=False) as f:
        e={k:f[k] for k in f.files}
    i=meta['index'];flat=e['snapshot_states'][i]
    task=benchmark.get_benchmark_dict()['libero_10']().get_task(meta['task_id'])
    env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,meta['seed'])
    results=[]
    try:
        env.reset()
        for mode in ['legacy','primary_reapply','warmstart_zero','forward_twice']:
            obs=restore_snapshot(env,flat,e['snapshot_metadata'][i])
            before=env.get_sim_state().copy()
            if mode=='primary_reapply':env.set_state(flat)
            if mode=='warmstart_zero':env.sim.data.qacc_warmstart[:]=0
            if mode=='forward_twice':env.sim.forward()
            after=env.get_sim_state().copy()
            diff=np.flatnonzero(before!=flat)
            errors=[]
            for c in range(i,min(i+15,len(e['rewards']))):
                for a in e['actions'][c][:int(e['lengths'][c])]:obs,_,_,_=env.step(a.tolist())
                if c+1<len(e['snapshot_states']):
                    p=policy_observation(obs,prompt)
                    errors.append({'after_chunk':c,'steps':int(e['lengths'][i:c+1].sum()),
                                   'state_error':float(np.max(np.abs(env.get_sim_state()-e['snapshot_states'][c+1]))),
                                   'pixel_max':float(max(np.max(np.abs(p['observation/image'].astype(float)-e['base'][c+1])),np.max(np.abs(p['observation/wrist_image'].astype(float)-e['wrist'][c+1]))))})
            result={'mode':mode,'initial_difference_indices':diff.tolist(),'initial_differences':(before-flat)[diff].tolist(),
                    'after_reapply_error':float(np.max(np.abs(after-flat))),'recorded_action_errors':errors}
            results.append(result);print('PRECISION_ABLATION',json.dumps(result),flush=True)
        src={}
        for label,obj in [('sim_set_state_from_flattened',env.sim.set_state_from_flattened),('sim_set_state',env.sim.set_state),('sim_forward',env.sim.forward),('environment_step',env.env.step),('controller_run',env.robots[0].controller.run_controller),('controller_update',env.robots[0].controller.update)]:
            try:src[label]={'file':inspect.getsourcefile(obj),'source':inspect.getsource(obj)}
            except Exception as error:src[label]={'error':str(error)}
        (ROOT/'artifacts/simulator_source_inventory.json').write_text(json.dumps(src,indent=2))
        (ROOT/'artifacts/restore_precision_ablation.json').write_text(json.dumps(results,indent=2))
    finally:env.close()

if __name__=='__main__':main()
