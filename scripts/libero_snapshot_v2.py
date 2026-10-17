"""Versioned bit-preserving primary state restore after derived-state forward.

Never changes environment/controller configuration, reward, or original datasets.
V1 snapshots require an explicit migration with na=0 proof; they are not silently
treated as V2. mj_forward normalizes free-joint quaternions by one ULP and its
derived-state pass can alter warm start. Saved primary and integration state is
reapplied AFTER that pass, while cached observation clocks remain unchanged.
"""
import json
from decimal import Decimal
import xml.etree.ElementTree as ET
import numpy as np
from libero_snapshot import take_snapshot as take_v1, restore_snapshot as restore_v1, array_state, array_restore, model_fingerprint

SNAPSHOT_VERSION=2

def recover_fixture_samples(env, meta, model_xml):
    """Recover exact RNG-generated poses, never rounded XML pose values.

    XML formatting intervals only identify candidate draws. The complete
    original model hash validates every restored finite-double value.
    """
    tree=ET.fromstring(str(model_xml));bodies={b.attrib['name']:b for b in tree.iter('body')}
    source=meta['numpy_rng'];proof=[]
    for name,sampler in env.env.placement_initializer.samplers.items():
        if not all(obj.name in env.env.fixtures_dict for obj in sampler.mujoco_objects):continue
        assert len(sampler.mujoco_objects)==1,'Multiple fixture sampler needs explicit reconstruction'
        obj=sampler.mujoco_objects[0];tokens=bodies[obj.root_body].attrib.get('pos','0 0 0').split()
        rounded=np.asarray([float(x) for x in tokens])
        intervals=np.asarray([float(Decimal(10)**Decimal(x).as_tuple().exponent)*.501 for x in tokens])
        matches={}
        for cursor in range(0,min(int(source[2])+2,624)):
            np.random.set_state((source[0],np.asarray(source[1],np.uint32),cursor,0,0.))
            for pos,quat,placed in sampler.sample().values():
                pos=np.asarray(pos,np.float64);quat=np.asarray(quat,np.float64)
                if placed.name==obj.name and np.all(np.abs(pos-rounded)<=intervals):
                    key=pos.tobytes()+quat.tobytes()
                    matches.setdefault(key,{'cursor':cursor,'position':pos,'quaternion':quat})
        if len(matches)!=1:
            raise ValueError('Exact fixture RNG candidate ambiguous/missing: '+name+' count='+str(len(matches)))
        candidate=next(iter(matches.values()));body=env.sim.model.body_name2id(obj.root_body)
        env.sim.model.body_pos[body]=candidate['position'];env.sim.model.body_quat[body]=candidate['quaternion']
        proof.append({'sampler':name,'body':obj.root_body,'cursor':candidate['cursor'],
                      'position':candidate['position'].tolist(),'quaternion':candidate['quaternion'].tolist(),
                      'xml_used_only_as_candidate_formatting_interval':True})
    if model_fingerprint(env)!=meta['model_fingerprint']:
        raise ValueError('Independent exact fixture RNG reconstruction fails full physical fingerprint')
    return proof

def recover_model_state(env, meta, model_xml=None):
    """Reconstruct omitted fixture placement; require exact full model hash.

    LIBERO writes randomized fixture poses into model.body_pos/body_quat,
    which are absent from qpos/qvel. Never reconstruct from rounded model XML.
    A V1 record has the complete MT19937 block, allowing placement draws to be
    replayed within that block. A match is proof, not an approximate fit.
    """
    expected=meta['model_fingerprint']
    initial=model_fingerprint(env)
    proof={'initial_fingerprint':initial,'expected_fingerprint':expected,
           'method':'fresh model already identical','attempts':0}
    if initial!=expected:
        if model_xml is not None:
            fixtures=recover_fixture_samples(env,meta,model_xml)
            proof.update(method='exact per-fixture MT19937 draws and full model hash',fixtures=fixtures)
        else:
            source=meta['numpy_rng'];end=min(int(source[2])+2,624)
            cursors=list(range(0,end,2))+list(range(1,end,2))
            found=False
            for cursor in cursors:
                np.random.set_state((source[0],np.asarray(source[1],np.uint32),cursor,0,0.))
                env.reset();proof['attempts']+=1
                if model_fingerprint(env)==expected:
                    proof.update(method='exact MT19937 placement replay',start_cursor=cursor,
                                 post_cursor=int(np.random.get_state()[2]))
                    found=True;break
            if not found:
                raise ValueError('Cannot exactly reconstruct omitted V1 fixture model: '+json.dumps(proof))
    proof['matched_fingerprint']=model_fingerprint(env)
    assert proof['matched_fingerprint']==expected
    saved={k:array_state(np.asarray(getattr(env.sim.model,k)).copy()) for k in ['body_pos','body_quat']}
    return saved,proof

def migrate_v1_for_branch(env, flat, metadata, model_xml=None):
    meta=json.loads(str(metadata))
    if meta.get('snapshot_version',1)!=1:
        raise ValueError('Explicit migration requires a version1 source')
    nq,nv,na=int(env.sim.model.nq),int(env.sim.model.nv),int(env.sim.model.na)
    if na!=0 or len(flat)!=1+nq+nv:
        raise ValueError('Legacy flattened snapshot cannot recover actuator activation')
    model_state,proof=recover_model_state(env,meta,model_xml)
    meta['snapshot_version']=2
    meta['mutable_model_state']=model_state
    meta['migration']={'from_version':1,'source_primary_state':'original immutable flattened state',
                       'act_reconstruction':'na=0, empty activation proven',
                       'missing_historical_fields':['qacc (derived); original per-step states outside chunk anchors'],
                       'requires_full_trajectory_reproduction_gate':True,
                       'model_reconstruction_proof':proof}
    meta['primary_state']={'time':float(flat[0]),'qpos':array_state(np.asarray(flat[1:1+nq]).copy()),
                           'qvel':array_state(np.asarray(flat[1+nq:]).copy()),'act':array_state(np.empty(0,np.float64))}
    return json.dumps(meta,separators=(',',':'))

def take_snapshot(env):
    flat,metadata=take_v1(env)
    meta=json.loads(metadata);meta['snapshot_version']=2
    data=env.sim.data
    meta['mutable_model_state']={k:array_state(np.asarray(getattr(env.sim.model,k)).copy()) for k in ['body_pos','body_quat']}
    meta['primary_state']={'time':float(data.time),**{k:array_state(np.asarray(getattr(data,k)).copy()) for k in ['qpos','qvel','act']}}
    meta['integration_state']={k:array_state(np.asarray(getattr(data,k)).copy()) for k in
        ['qacc','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat'] if hasattr(data,k)}
    return flat,json.dumps(meta,separators=(',',':'))

def restore_snapshot(env,flat,metadata):
    meta=json.loads(str(metadata))
    if meta.get('snapshot_version')!=2:
        raise ValueError('V2 restore refuses unversioned/V1 snapshot; migrate explicitly')
    if 'mutable_model_state' not in meta:
        raise ValueError('V2 snapshot is missing exact mutable fixture model state')
    for key,saved in meta['mutable_model_state'].items():
        target=getattr(env.sim.model,key);value=array_restore(saved)
        if target.shape!=value.shape:raise ValueError('Mutable model shape mismatch: '+key)
        target[:]=value
    # Legacy restore checks the complete physical model fingerprint before
    # applying primary/controller/cache state. No physics tolerance is relaxed.
    obs=restore_v1(env,flat,metadata)
    # Preserve the original finite double values instead of the normalized values
    # written by mj_forward. No second forward here: that would repeat mutation.
    primary=meta['primary_state'];env.sim.data.time=primary['time']
    for key in ['qpos','qvel','act']:
        target=getattr(env.sim.data,key)
        saved=array_restore(primary[key])
        if target.shape!=saved.shape:raise ValueError('Primary state shape mismatch: '+key)
        target[:]=saved
    for key,saved in meta.get('integration_state',{}).items():
        target=getattr(env.sim.data,key);target[:]=array_restore(saved)
    if not np.array_equal(np.asarray(env.get_sim_state()),np.asarray(flat)):
        raise AssertionError('V2 primary state is not bit identical to original flat state')
    return obs
