# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Exact fixture poses from sampler RNG, using rounded XML only to filter.

The rounded XML is never loaded as a model or used as the restored pose.
The final complete physical fingerprint must be exactly the recorded hash.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from libero_snapshot import model_fingerprint

ROOT=Path('/opt/divl-trqam')

def main():
    p=argparse.ArgumentParser();p.add_argument('--state-id',default='task03_init015_rep0_chunk026');args=p.parse_args()
    item=next(v for v in json.loads((ROOT/'artifacts/static_action_scalar_step2000.json').read_text())['records'] if v['state_id']==args.state_id)
    with np.load(item['episode_path'],allow_pickle=False) as f:
        meta=json.loads(str(f['snapshot_metadata'][item['index']]));xml=ET.fromstring(str(f['model_xml']))
    poses={b.attrib['name']:np.fromstring(b.attrib.get('pos','0 0 0'),sep=' ') for b in xml.iter('body')}
    task=benchmark.get_benchmark_dict()['libero_10']().get_task(item['task_id'])
    env,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,item['seed']);env.reset()
    results=[];source=meta['numpy_rng'];d=env.env
    try:
        print('SAMPLER_ORDER',list(d.placement_initializer.samplers),flush=True)
        for name,sampler in d.placement_initializer.samplers.items():
            objects=sampler.mujoco_objects
            if not all(obj.name in d.fixtures_dict for obj in objects):continue
            print('SAMPLER',name,repr(vars(sampler)),flush=True)
            matches=[]
            for cursor in range(0,min(int(source[2])+2,624)):
                np.random.set_state((source[0],np.asarray(source[1],np.uint32),cursor,0,0.))
                placements=sampler.sample()
                for obj_pos,obj_quat,obj in placements.values():
                    target=poses[obj.root_body]
                    error=float(np.max(np.abs(np.asarray(obj_pos)-target)))
                    if error<=1e-6:
                        row={'sampler':name,'object':obj.name,'body':obj.root_body,'cursor':cursor,
                             'position':np.asarray(obj_pos).tolist(),'quaternion':np.asarray(obj_quat).tolist(),
                             'xml_filter_error':error}
                        matches.append(row);print('CANDIDATE',json.dumps(row),flush=True)
            assert len(matches)==1,('Fixture reconstruction ambiguous',name,len(matches))
            match=matches[0];body=env.sim.model.body_name2id(match['body'])
            env.sim.model.body_pos[body]=match['position'];env.sim.model.body_quat[body]=match['quaternion']
            results.append(match)
        actual=model_fingerprint(env)
        result={'state_id':args.state_id,'fixtures':results,'expected':meta['model_fingerprint'],'actual':actual,'passed':actual==meta['model_fingerprint']}
        (ROOT/'artifacts'/('fixture_exact_recovery_'+args.state_id+'.json')).write_text(json.dumps(result,indent=2))
        print('EXACT_FIXTURE_RECOVERY',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':main()
