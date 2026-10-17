# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Replay only the selected legal Train50; recover real next observations.

No policy training, no new demonstration or deployment collection. Sources are
read-only. Each task has its own immutable output and failures remain preserved.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys, time, traceback
ROOT = Path('/data/divl-trqam/benchmark')
EXP = ROOT/'experiments/lwd_expert_rollout_mix_v1'
sys.path.insert(0, str(ROOT/'scripts'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def write(p,j):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
    a=argparse.ArgumentParser();a.add_argument('--task-id',type=int,required=True);args=a.parse_args();tid=args.task_id
    import h5py, numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from native_action_audit_v1 import native_action_contract,passive_action_record
    from benchmark_data_v1 import normalize
    p=json.loads((EXP/'configs/protocol001.json').read_text());old=p['inherited']
    assert sha(old['train_manifest'])==old['train_manifest_sha256']
    assert sha(old['normalization'])==old['normalization_sha256']
    m=json.loads(Path(old['train_manifest']).read_text());norm=json.loads(Path(old['normalization']).read_text())
    original=json.loads((ROOT/'data/expert/expert_manifest.json').read_text())
    es=[e for e in m['episodes'] if e['task_id']==tid];assert len(es)==5
    assert sha(es[0]['path'])==old['native_source_files_sha256'][tid]
    dest=EXP/'data/performance2/expert_transitions'/f'task_{tid:02d}.hdf5'
    result=EXP/'metrics/expert_audit'/f'task_{tid:02d}.json'
    start=time.monotonic();rows=[];env=None
    out={'task_id':tid,'passed':False,'scientific_updates':0,'scope':'replay original selected Train50 actions only','pid':os.getpid()}
    try:
        env=OffScreenRenderEnv(bddl_file_name=original['tasks'][tid]['bddl_path'],camera_heights=128,camera_widths=128,render_gpu_device_id=0)
        env.seed(20261009+tid)
        with h5py.File(es[0]['path'],'r') as src,h5py.File(dest,'x') as dst:
            dst.attrs['source_sha256']=old['native_source_files_sha256'][tid]
            for e in es:
                g=src[e['demo']];n=e['length'];assert len(g['actions'])==n
                assert e['native_success'] and e['expert_training_eligible']
                assert g.attrs['state_source_id']==e['state_source_id']
                state0=g['initial_state'][:];assert hashlib.sha256(state0.tobytes()).hexdigest()==e['initial_state_sha256']
                env.reset();env.reset_from_xml_string(str(g.attrs['relocated_model_xml']));env.sim.reset();obs=env.set_init_state(state0)
                initial_error=float(np.max(np.abs(env.get_sim_state()-state0)));assert initial_error<1e-12
                contract=native_action_contract(env)
                ep='expert:'+e['unique_episode_id'];h=dst.create_group(ep);ob=h.create_group('observations')
                for k,shape,dtype in [('agentview_rgb',(128,128,3),'u1'),('eye_in_hand_rgb',(128,128,3),'u1'),('proprio',(8,),'f4'),('sim_state',(len(state0),),'f8')]:
                    ob.create_dataset(k,(n+1,*shape),dtype=dtype,chunks=(1,*shape),compression='lzf')
                rewards=[];dones=[];success=[];state_error=0.;proprio_error=0.;pixel_error=0;action_error=0.;bound_violations=0
                def save_obs(i):
                    ob['agentview_rgb'][i]=obs['agentview_image'];ob['eye_in_hand_rgb'][i]=obs['robot0_eye_in_hand_image']
                    ob['proprio'][i]=np.concatenate([obs['robot0_eef_pos'],quat2axisangle(np.asarray(obs['robot0_eef_quat']).copy()),obs['robot0_gripper_qpos']])
                    ob['sim_state'][i]=env.get_sim_state()
                for i in range(n):
                    save_obs(i)
                    se=float(np.max(np.abs(ob['sim_state'][i]-g['states'][i])));state_error=max(state_error,se);assert se<1e-9,('state_alignment',ep,i,se)
                    pe=float(np.max(np.abs(ob['proprio'][i]-g['proprio'][i])));proprio_error=max(proprio_error,pe);assert pe<1e-6
                    for k in ['agentview_rgb','eye_in_hand_rgb']:
                        px=int(np.max(np.abs(ob[k][i].astype(np.int16)-g[k][i].astype(np.int16))));pixel_error=max(pixel_error,px)
                        assert px==0,('rgb_alignment',ep,i,k,px)
                    action=np.asarray(g['actions'][i],np.float64);assert action.shape==(7,) and np.isfinite(action).all()
                    record=passive_action_record(contract,action);bound_violations+=record['input_bound_violations']
                    # Execute original float64 actions, without clipping or reward fabrication.
                    obs,r,d,info=env.step(action.tolist());s=bool(env.check_success());rewards.append(float(r));dones.append(bool(d));success.append(s)
                    assert float(r)==float(g['rewards'][i]) and bool(d)==bool(g['dones'][i]),('native_reward_done',ep,i)
                    assert i==n-1 or not (s or d),('early_episode_boundary',ep,i)
                save_obs(n);assert success[-1] and sum(rewards)==e['native_return']
                raw=np.asarray(g['actions'][:],np.float32);normalized=normalize(raw,norm['statistics']['actions'])
                stats=norm['statistics']['actions'];lo=np.asarray(stats['q01'],np.float32);hi=np.asarray(stats['q99'],np.float32)
                denormalized=(normalized+1)/2*(hi-lo+np.float32(1e-6))+lo
                action_error=float(np.max(np.abs(raw-denormalized)));assert action_error<2e-6
                h.create_dataset('executed_actions',data=g['actions'][:]);h.create_dataset('native_rewards',data=np.asarray(rewards,np.float32));h.create_dataset('native_dones',data=dones);h.create_dataset('goal_success',data=success)
                # The original Critic requires full fixed H5. Expert terminal tails
                # have no issued plan; exclude them instead of inventing actions.
                starts=np.arange(0,n-4,5,dtype=np.int32);ends=starts+5
                ch=h.create_group('chunks');ch.create_dataset('start_observation_index',data=starts);ch.create_dataset('next_observation_index',data=ends)
                ch.create_dataset('actual_k',data=np.full(len(starts),5,np.int32))
                ch.create_dataset('planned_h5_normalized',data=np.stack([normalized[i:i+5] for i in starts]));ch.create_dataset('planned_h5_fixed_mask',data=np.ones((len(starts),5,7),np.float32))
                ch.create_dataset('chunk_reward_discounted',data=[sum(rewards[i+j]*.99**j for j in range(5)) for i in starts],dtype='f4')
                terminal=[bool(any(dones[i:i+5]) or any(success[i:i+5])) for i in starts]
                ch.create_dataset('bootstrap_discount',data=[0. if t else .99**5 for t in terminal],dtype='f4');ch.create_dataset('native_terminal',data=[any(dones[i:i+5]) for i in starts]);ch.create_dataset('goal_success',data=[any(success[i:i+5]) for i in starts])
                ba=np.zeros((len(starts),10,32),np.float32);bm=np.zeros_like(ba);eligible=[]
                for j,i in enumerate(starts):
                    k=min(10,n-i);ba[j,:k,:7]=normalized[i:i+k];bm[j,:k,:7]=1;eligible.append(k==10)
                ch.create_dataset('bc_actual_h10_normalized',data=ba);ch.create_dataset('bc_actual_h10_mask',data=bm);ch.create_dataset('bc_full_h10_eligible',data=eligible)
                h.attrs.update(state_source_id=e['state_source_id'],collection_censored=False,action_semantics='expert executed behavior H5; not pi05 issued plan',task_id=tid,language=str(g.attrs['language']))
                row={**e,'episode_id':ep,'success':True,'trajectory_path':str(dest),'transitions':len(starts),'source':'expert','executed_steps':n,'initial_error':initial_error,'state_max_error':state_error,'proprio_max_error':proprio_error,'rgb_max_error':pixel_error,'normalization_roundtrip_max_error':action_error,'native_reward_sum':sum(rewards),'native_terminal':dones[-1],'goal_success':success[-1],'excluded_partial_h5_chunks':int(n%5!=0),'excluded_tail_actions':n%5,'exclusion_reason':'no full executed expert H5 exists at terminal remainder; no fake plan/padding','bound_violations_native_handled':bound_violations,'raw_action_dim':7,'full_h5_count':len(starts)}
                rows.append(row);dst.flush();write(result,{**out,'completed_episodes':len(rows),'records':rows,'complete':False});print('EXPERT_TRANSITION_AUDITED',ep,n,len(starts),flush=True)
        out.update(passed=True,complete=True,records=rows,output=str(dest),output_sha256=sha(dest),output_bytes=dest.stat().st_size,completed_episodes=5)
    except Exception:
        out.update(complete=False,records=rows,error=traceback.format_exc());print(out['error'],flush=True)
    finally:
        if env:env.close()
        out.update(wall_s=time.monotonic()-start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(result,out)
    sys.exit(0 if out['passed'] else 1)
if __name__=='__main__':main()
