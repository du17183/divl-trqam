"""Explicit simulator/controller snapshot for diagnostic branches, not actor input."""
import json
import base64
import hashlib
import random
import zlib
import numpy as np


def array_state(value):
    result={'dtype':str(value.dtype),'shape':list(value.shape)}
    if value.nbytes>4096:result['blob']=base64.b64encode(zlib.compress(value.tobytes(),1)).decode('ascii')
    else:result['array']=value.tolist()
    return result

def array_restore(value):
    if 'blob' in value:return np.frombuffer(zlib.decompress(base64.b64decode(value['blob'])),dtype=value['dtype']).reshape(value['shape']).copy()
    return np.asarray(value['array'],dtype=value['dtype']).reshape(value['shape'])

def numeric_state(obj):
    result={}
    for key,value in vars(obj).items():
        if isinstance(value,np.ndarray):result[key]=array_state(value)
        elif isinstance(value,(bool,int,float,str)) or value is None:result[key]={'scalar':value}
        elif isinstance(value,np.generic):result[key]={'scalar':value.item()}
    return result


def apply_numeric(obj,values):
    for key,value in values.items():
        setattr(obj,key,array_restore(value) if 'array' in value or 'blob' in value else value['scalar'])


def model_fingerprint(env):
    h=hashlib.sha256()
    for key in ['body_mass','body_inertia','body_pos','body_quat','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','dof_armature','dof_damping','jnt_range','actuator_gainprm','actuator_biasprm']:
        value=np.asarray(getattr(env.sim.model,key));h.update(key.encode());h.update(value.tobytes())
    h.update(str(env.env.control_freq).encode())
    return h.hexdigest()

def take_snapshot(env):
    data=env.sim.data
    hidden={k:np.asarray(getattr(data,k)).copy().tolist() for k in ['qacc_warmstart','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat'] if hasattr(data,k)}
    controllers=[]
    for robot in env.robots:
        controller=robot.controller
        row={'controller':numeric_state(controller),'gripper':numeric_state(robot.gripper)}
        for k in ['interpolator_pos','interpolator_ori']:
            obj=getattr(controller,k,None)
            if obj is not None:row[k]=numeric_state(obj)
        controllers.append(row)
    numpy_rng=np.random.get_state()
    observables={key:numeric_state(value) for key,value in env.env._observables.items()}
    cache={key:({'observable':key} if key in observables else array_state(np.asarray(value))) for key,value in env.env._obs_cache.items()}
    metadata={'model_fingerprint':model_fingerprint(env),'physics_arrays':hidden,'controllers':controllers,'observables':observables,'observation_cache':cache,
        'env_counters':{k:getattr(env.env,k) for k in ['timestep','cur_time','done'] if hasattr(env.env,k)},
        'numpy_rng':[numpy_rng[0],numpy_rng[1].tolist(),int(numpy_rng[2]),int(numpy_rng[3]),float(numpy_rng[4])],
        'python_rng':random.getstate()}
    return np.asarray(env.get_sim_state()).copy(),json.dumps(metadata,separators=(',',':'))


def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,list) else x


def restore_snapshot(env,flat,metadata):
    meta=json.loads(str(metadata))
    if model_fingerprint(env)!=meta['model_fingerprint']:raise ValueError('Official model physical constants differ from snapshot')
    env.set_state(np.asarray(flat));env.sim.forward()
    for key,value in meta['physics_arrays'].items():
        target=getattr(env.sim.data,key)
        target[:]=np.asarray(value).reshape(target.shape)
    for robot,row in zip(env.robots,meta['controllers'],strict=True):
        apply_numeric(robot.controller,row['controller']);apply_numeric(robot.gripper,row['gripper'])
        for key in ['interpolator_pos','interpolator_ori']:
            if key in row:apply_numeric(getattr(robot.controller,key),row[key])
    for key,value in meta['env_counters'].items():setattr(env.env,key,value)
    state=meta['numpy_rng'];np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))
    random.setstate(tuples(meta['python_rng']))
    # Preserve sensor sampling clocks and cached images. A forced sensor update
    # changes the observation age and can change the next policy input even
    # when MuJoCo qpos/qvel and all later action-only states are identical.
    if set(meta['observables'])!=set(env.env._observables):raise ValueError('Observable config mismatch')
    for key,value in meta['observables'].items():apply_numeric(env.env._observables[key],value)
    env.env._obs_cache={key:(env.env._observables[value['observable']]._current_observed_value.copy()
        if 'observable' in value else array_restore(value)) for key,value in meta['observation_cache'].items()}
    return env.env._get_observations()
