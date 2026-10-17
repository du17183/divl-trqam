import numpy as np,json
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION,LIBERO_DUMMY_ACTION
from libero_snapshot import take_snapshot,restore_snapshot
suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(0)
env,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,20261007)
env.reset();env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(10):env.step(LIBERO_DUMMY_ACTION)
flat,hidden=take_snapshot(env);xml=env.sim.model.get_xml()
other,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,20261007);other.reset();other.reset_from_xml_string(xml);restore_snapshot(other,flat,hidden)
for key in dir(env.sim.model):
    try:
        a=getattr(env.sim.model,key);b=getattr(other.sim.model,key)
        if isinstance(a,np.ndarray) and a.dtype.kind in 'fiu' and a.size and a.shape==b.shape:
            delta=float(np.max(np.abs(a.astype(float)-b.astype(float))))
            if delta>0:print('MODEL_DIFF',key,a.shape,delta,flush=True)
    except Exception as ex:pass
for i,(a,b) in enumerate(zip(env.robots,other.robots)):
    for key in vars(a):
        va=getattr(a,key);vb=getattr(b,key,None)
        if isinstance(va,np.ndarray) and isinstance(vb,np.ndarray) and va.shape==vb.shape and va.size:
            d=float(np.max(np.abs(va.astype(float)-vb.astype(float))))
            if d:print('ROBOT_DIFF',key,d,flush=True)
env.close();other.close()
