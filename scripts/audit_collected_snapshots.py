# Public release: deployment paths anonymized; algorithm equations unchanged.
import json,numpy as np
from pathlib import Path
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from libero_snapshot import restore_snapshot
r=Path('/opt/divl-trqam');suite=benchmark.get_benchmark_dict()['libero_10']();results=[]
for task_id in [0,3,5]:
    path=sorted((r/'data/pilot_libero_v1/episodes').glob(f'task{task_id:02d}_*.npz'))[0]
    with np.load(path,allow_pickle=False) as f:e={k:f[k] for k in f.files}
    task=suite.get_task(task_id);env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,int(e['seed']))
    try:
        env.reset()
        for i in sorted(set([len(e['rewards'])//3,2*len(e['rewards'])//3])):
            obs=restore_snapshot(env,e['snapshot_states'][i],e['snapshot_metadata'][i])
            before=policy_observation(obs,prompt)
            state_error=float(np.max(np.abs(before['observation/state']-e['state'][i])))
            pixel_error=float(np.max(np.abs(before['observation/image'].astype(float)-e['base'][i].astype(float))))
            for action in e['actions'][i,e['action_mask'][i]]:obs,_,_,_=env.step(action.tolist())
            next_error=float(np.max(np.abs(env.get_sim_state()-e['snapshot_states'][i+1])))
            after=policy_observation(obs,prompt)
            next_pixel_error=float(np.max(np.abs(after['observation/image'].astype(float)-e['base'][i+1].astype(float))))
            assert state_error<1e-5 and pixel_error==0 and next_error<1e-7 and next_pixel_error<=1,(state_error,pixel_error,next_error,next_pixel_error)
            row={'episode':path.name,'chunk':i,'task':task_id,'state_error':state_error,'pixel_error':pixel_error,'next_state_error':next_error,'next_pixel_error':next_pixel_error}
            results.append(row);print('COLLECTED_SNAPSHOT',json.dumps(row),flush=True)
    finally:env.close()
(r/'artifacts/collected_snapshot_audit.json').write_text(json.dumps({'passed':True,'tests':results},indent=2))
