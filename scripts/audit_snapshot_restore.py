# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Require repeatable real LIBERO state/action branches before collecting pilot."""
from pathlib import Path
import json
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION
from libero_snapshot import take_snapshot, restore_snapshot

ROOT=Path('/opt/divl-trqam')

def rollout(env,actions):
    values=[]
    for a in actions:
        obs,r,d,_=env.step(a.tolist())
        values.append((env.get_sim_state().copy(),float(r),bool(d)))
    return obs,values

def main():
    assert json.loads((ROOT/'artifacts/four_arm_smoke_test.json').read_text())['passed']
    split=json.loads((ROOT/'configs/splits/libero_factorial_v1.json').read_text())
    suite=benchmark.get_benchmark_dict()['libero_10']();results=[]
    rng=np.random.default_rng(20261007)
    for task_id in [0,4,8]:
        task=suite.get_task(task_id);init=split['tasks'][task.name]['train'][0]
        env,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,20261007+task_id)
        try:
            env.reset();env.set_init_state(suite.get_task_init_states(task_id)[init])
            for _ in range(10):env.step(LIBERO_DUMMY_ACTION)
            actions=rng.uniform(-.1,.1,(5,7));actions[:,-1]=-1
            flat,hidden=take_snapshot(env);xml=env.sim.model.get_xml()
            baseline,expected=rollout(env,actions)
            for repetition in range(3):
                restore_snapshot(env,flat,hidden)
                initial_error=float(np.max(np.abs(env.get_sim_state()-flat)))
                obs,actual=rollout(env,actions)
                step_error=max(float(np.max(np.abs(a[0]-b[0]))) for a,b in zip(expected,actual,strict=True))
                pixel_error=max(float(np.max(np.abs(obs[k].astype(float)-baseline[k].astype(float)))) for k in ['agentview_image','robot0_eye_in_hand_image'])
                assert initial_error<1e-9 and step_error<1e-7 and pixel_error<=1, (initial_error,step_error,pixel_error)
                assert [(a[1],a[2]) for a in actual]==[(a[1],a[2]) for a in expected]
                results.append(dict(task_id=task_id,initial_state_id=init,repeat=repetition,initial_error=initial_error,step_error=step_error,pixel_error=pixel_error,recreated=False))
            second,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,20261007+task_id)
            try:
                second.reset()
                restore_snapshot(second,flat,hidden)
                initial_error=float(np.max(np.abs(second.get_sim_state()-flat)))
                obs,actual=rollout(second,actions)
                step_error=max(float(np.max(np.abs(a[0]-b[0]))) for a,b in zip(expected,actual,strict=True))
                pixel_error=max(float(np.max(np.abs(obs[k].astype(float)-baseline[k].astype(float)))) for k in ['agentview_image','robot0_eye_in_hand_image'])
                assert initial_error<1e-9 and step_error<1e-7 and pixel_error<=1,(initial_error,step_error,pixel_error)
                results.append(dict(task_id=task_id,initial_state_id=init,initial_error=initial_error,step_error=step_error,pixel_error=pixel_error,recreated=True))
            finally:second.close()
        finally:env.close()
    result={'passed':True,'tests':results,'scope':'Three tasks, same simulator and recreated official task environment, physical-constant fingerprint, observation clocks/cache, controller and warm-start state'}
    (ROOT/'artifacts/pilot_snapshot_preflight.json').write_text(json.dumps(result,indent=2))
    print('SNAPSHOT_AUDIT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
