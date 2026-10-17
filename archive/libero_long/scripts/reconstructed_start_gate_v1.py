"""Native physical-start reproducibility. No new expert collection or policy training."""
from pathlib import Path
import datetime
import hashlib
import json
import time
import h5py
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

ROOT=Path(__file__).resolve().parents[1]


def restore(env, group):
    env.reset();env.reset_from_xml_string(str(group.attrs['relocated_model_xml']));env.sim.reset()
    initial=group['initial_state'][:]
    obs=env.set_init_state(initial)
    assert np.max(np.abs(env.get_sim_state()-initial))<1e-12
    return obs


def main():
    start=time.monotonic();rows=[]
    split=json.loads((ROOT/'configs/libero_10_split_manifest.json').read_text())
    original=json.loads((ROOT/'data/expert/expert_manifest.json').read_text())
    suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
    for task_id in range(10):
        task=suite.get_task(task_id);assert task.name==split['tasks'][task_id]['task_name']
        e=split['tasks'][task_id]['partitions']['dev'][0]
        env=OffScreenRenderEnv(bddl_file_name=original['tasks'][task_id]['bddl_path'],camera_heights=128,camera_widths=128,render_gpu_device_id=0)
        env.seed(20261009+task_id)
        try:
            with h5py.File(ROOT/'data/expert/native_aligned_v2'/('task_%02d.hdf5'%task_id),'r') as h:
                group=h[e['demo']]; traces=[];images=[]
                assert group.attrs['state_source_id']==e['state_source_id']
                assert hashlib.sha256(group['initial_state'][:].tobytes()).hexdigest()==e['initial_state_sha256']
                for repeat in range(2):
                    obs=restore(env,group);images.append(obs['agentview_image'].copy());trace=[]
                    for action in group['actions'][:5]:
                        trace.append(env.get_sim_state().copy());obs,reward,done,_=env.step(action.tolist())
                        if done:break
                    traces.append(np.stack(trace))
                assert traces[0].shape==traces[1].shape
                error=float(np.max(np.abs(traces[0]-traces[1])));assert error<1e-12
                rgb_error=int(np.max(np.abs(images[0].astype(np.int16)-images[1].astype(np.int16))));assert rgb_error==0
                derived_error=float(np.max(np.abs(traces[0]-group['states'][:len(traces[0])])));assert derived_error<1e-9
                rows.append({'task_id':task_id,'state_source_id':e['state_source_id'],'repeat_state_max_error':error,
                             'repeat_rgb_max_error':rgb_error,'derived_record_state_max_error':derived_error,
                             'initial_time':float(group['initial_state'][0]),'extra_settle_steps':0})
        finally:env.close()
        print('NATIVE_START_REPRODUCED',rows[-1],flush=True)
    result={'passed':True,'synthetic':False,'rows':rows,'wall_s':time.monotonic()-start,
            'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'reconstructed_start_settle_steps':0,'scope':'all10 tasks, first fixed Dev source, two identical first5 expert-prefix replays; no Final; no success tuning',
            'standard_protocol_settle_steps':10,'standard_train_source_overlap':'unknown'}
    with (ROOT/'metrics/reconstructed_start_gate001.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
