from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from collect_action_sensitive_replay_v3 import phase_features
import numpy as np,json
env,prompt=_get_libero_env(benchmark.get_benchmark_dict()['libero_10']().get_task(0),LIBERO_ENV_RESOLUTION,2026100800)
try:
 env.seed(2026100800);env.reset()
 obs=env.set_init_state(benchmark.get_benchmark_dict()['libero_10']().get_task_init_states(0)[0])
 f=phase_features(env,policy_observation(obs,prompt),np.zeros((10,7),np.float32),None)
 assert f['object_count']>0
 print('V3_PHYSICAL_SELECTION_PROBE',json.dumps(f),flush=True)
finally:env.close()
