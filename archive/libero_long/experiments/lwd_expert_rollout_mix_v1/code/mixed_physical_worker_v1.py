"""Same full native prefix, bounded counterfactuals; diagnostic data never trains."""

from pilot8_common_v1 import *
from mixed_common_v2 import EXP,CODE,OLD

import argparse

import h5py

import numpy as np

from multiprocessing.connection import Client





def raw_obs(obs):

    from robosuite.utils.transform_utils import quat2axisangle

    return {'agentview_rgb':np.asarray(obs['agentview_image'],np.uint8),

            'eye_in_hand_rgb':np.asarray(obs['robot0_eye_in_hand_image'],np.uint8),

            'proprio':np.concatenate([obs['robot0_eef_pos'],quat2axisangle(np.asarray(obs['robot0_eef_quat']).copy()),obs['robot0_gripper_qpos']]).astype(np.float32)}





def hidden_state(env):

    """Compare generated physics warm-start/actuation and controller goal history."""

    d=env.sim.data

    values={}

    for key in ['qacc_warmstart','ctrl','act','qpos','qvel','mocap_pos','mocap_quat']:

        if hasattr(d,key):values['physics_'+key]=np.asarray(getattr(d,key)).copy()

    controller=env.env.robots[0].controller

    for key in ['goal_pos','goal_ori','initial_joint','ee_pos','ee_ori_mat']:

        if hasattr(controller,key) and getattr(controller,key) is not None:values['controller_'+key]=np.asarray(getattr(controller,key)).copy()

    robot=env.env.robots[0]

    if hasattr(robot.gripper,'current_action'):values['gripper_action']=np.asarray(robot.gripper.current_action).copy()

    return values





def main():

    p=argparse.ArgumentParser();p.add_argument('--case-id',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--socket',required=True);a=p.parse_args()

    from mixed_common_v2 import protected_audit,verify_sources
    verify_sources();protected_audit()
    for rel,h in read(EXP/'provenance/mixed_diagnostics_sources015.json')['files'].items():assert sha(EXP/rel)==h,rel
    verify_worker();budget=read(EXP/'configs/mixed_physical_validation015.json')

    case=budget['cases'][a.case_id];a.task_id=case['task_id'];assert case['case_id']==a.case_id

    from libero.libero.envs import OffScreenRenderEnv

    from reconstructed_start_gate_v1 import restore

    from replay_schema_v1 import normalize_real

    original=read(ROOT/'data/expert/expert_manifest.json');norm=read(config()['normalization']);stats=norm['statistics']['actions']

    low=np.asarray(stats['q01'],np.float32)[:7];high=np.asarray(stats['q99'],np.float32)[:7];scale=(high-low+np.float32(1e-6))/2

    path=EXP/f'metrics/mixed_physical015/case_{a.case_id:03d}.json';assert not path.exists()

    began=time.monotonic();result={'task_id':a.task_id,'task_name':case['task_name'],'case':case,'complete':False,'passed':False,'rows':[],

           'scientific_actor_updates':0,'scientific_critic_updates':0,'frozen_policy_continuation':True,

           'scope':'Diagnostic heldoutTrain source, exact recorded prefix; native sparse reward/gamma.99; no trainingReplay writes'}

    env=OffScreenRenderEnv(bddl_file_name=original['tasks'][a.task_id]['bddl_path'],camera_heights=128,camera_widths=128,render_gpu_device_id=a.gpu)

    try:

        with h5py.File(case['trajectory_path'],'r') as f,Client(a.socket,family='AF_UNIX',authkey=AUTHKEY) as client:

            g=f[case['episode_id']];ci=case['chunk_index'];start=int(g['chunks/start_observation_index'][ci]);original_state=g['observations/sim_state'][start]

            def prefix():

                env.seed(int(g.attrs['environment_seed']));obs=restore(env,g)

                for action in g['environment_actions_raw'][:start]:obs,_,done,_=env.step(action.tolist());assert not done,'Prefix reached native terminal'

                return obs

            obs=prefix();state=env.get_sim_state().copy();error=float(np.max(np.abs(state-original_state)));reference_hidden=hidden_state(env)

            result['recorded_prefix_state_max_absolute_error']=error

            if error>budget['recorded_prefix_absolute_tolerance']:

                result.update(complete=True,reason='Recorded state not reproducible with full executed native prefix; no valid physical pair',valid_states=0,valid_pairs=0);write(path,result,True);return

            raw=raw_obs(obs);client.send({'command':'directions','episode_id':case['episode_id'],'chunk_index':ci,'raw':raw});reply=client.recv();assert reply['ok'],reply

            from benchmark_io_v1 import unnormalize_actions

            base=np.asarray(reply['baseline_normalized_h5'],np.float32)

            full_padded=np.zeros((10,32),np.float32);full_padded[:5,:7]=base

            raw_base=unnormalize_actions(full_padded,norm)[:5]

            assert np.max(np.abs(raw_base-g['predicted_raw_h10'][ci,:5]))<1e-6,'Baseline action denormalization mismatch'

            rng=np.random.default_rng(np.random.SeedSequence([20261012,a.case_id,8106]))

            # Ranking candidates are generated without looking at either Q or its gradient.

            candidates=[('baseline',raw_base,None)]

            direction=rng.standard_normal(base.shape).astype(np.float32);direction[:,6]=0

            direction/=float(np.sqrt(np.mean(direction[:,:6]**2)))

            for label,size in [('independent_small',.03),('independent_large',.10)]:

                candidates.append((label,raw_base+direction*size,None))

            timing=raw_base.copy();timing[:,6]=np.roll(timing[:,6],2)

            if np.max(np.abs(timing[:,6]-raw_base[:,6]))<1e-8:timing[:,6]=-raw_base[:,6]

            candidates.append(('independent_gripper',timing,None))

            controller=env.env.robots[0].controller

            gripper=env.env.robots[0].gripper

            gripper_origin=reference_hidden.get('gripper_action',np.zeros(gripper.dof)).copy()

            def effective(actions):

                arm=np.stack([controller.scale_action(np.asarray(x[:6])).copy() for x in actions])

                values=[];gripper.current_action=gripper_origin.copy()

                for x in actions:values.append(np.asarray(gripper.format_action(np.asarray([x[6]]))).copy())

                gripper.current_action=gripper_origin.copy()

                return np.concatenate([arm,np.stack(values)],axis=1)

            base_effect=effective(raw_base)

            if case['gradient_selected']:

                target=float(budget['effective_controller7d_rms'])

                def matched(direction):

                    # Match after native arm clip/scaling AND gripper's actual discrete mapping.

                    direction=np.asarray(direction,np.float32);direction/=max(float(np.sqrt(np.mean(direction**2))),1e-12)

                    choices=[]

                    for scale0 in np.geomspace(1e-5,.5,120):

                        actions=raw_base+direction*scale0

                        rms=float(np.sqrt(np.mean((effective(actions)-base_effect)**2)))

                        choices.append((abs(rms-target),actions,rms))

                    return min(choices,key=lambda x:x[0])[1]

                for m in ['scalar','divl']:

                    direction=np.asarray(reply['gradients'][m],np.float32)*scale

                    assert np.isfinite(direction).all() and np.linalg.norm(direction)>0

                    for label,sign in [('positive',1),('negative',-1)]:

                        candidates.append((m+'_'+label,matched(direction*sign),m))

                random=rng.standard_normal(base.shape).astype(np.float32)

                candidates.append(('matched_random',matched(random),'random'))

            candidates=[x for x in candidates if x[0] in ('scalar_positive','divl_positive')]
            assert case['gradient_selected'] and len(candidates)==2
            origin_effect=base_effect.copy()

            for name,actions,method in candidates:

                obs=prefix();branch_state=env.get_sim_state();branch_hidden=hidden_state(env)

                assert branch_hidden.keys()==reference_hidden.keys()

                hidden_error=max((float(np.max(np.abs(branch_hidden[k]-reference_hidden[k]),initial=0)) for k in reference_hidden),default=0.)

                branch_error=float(np.max(np.abs(branch_state-state)))

                if max(hidden_error,branch_error)>budget['branch_identity_absolute_tolerance']:

                    result['rows'].append({'candidate':name,'valid':False,'branch_state_error':branch_error,'hidden_state_error':hidden_error,'reason':'Contact/controller history not reproducible'});continue

                normactions=normalize_real(actions,norm)

                client.send({'command':'score','episode_id':case['episode_id'],'chunk_index':ci,'raw':raw,'normalized_h5':normactions});q=client.recv();assert q['ok'],q

                rw=[];done=False;success=False;steps=start

                controller=env.env.robots[0].controller

                projected=effective(actions)

                if name=='baseline':origin_effect=projected.copy()

                effective_rms=float(np.sqrt(np.mean((projected-origin_effect)**2))) if origin_effect is not None else None

                trace=[]

                for action in actions:

                    obs,r,done,_=env.step(action.tolist());rw.append(float(r));steps+=1;success=bool(env.check_success());trace.append(env.get_sim_state().copy())

                    if done or success:break

                chunk=ci+1

                while not(done or success) and steps<config()['replay_max_steps']:

                    words=np.frombuffer(bytes.fromhex(case['state_source_id'])[:16],dtype='<u4')

                    noise=np.random.default_rng(np.random.SeedSequence([config()['replay_noise_seed'],0,*map(int,words),chunk])).standard_normal((10,32)).astype(np.float32)

                    ob=raw_obs(obs);client.send({'rgb':{k:ob[k] for k in ['agentview_rgb','eye_in_hand_rgb']},'proprio':ob['proprio'],'language':case['language'],'noise':noise});pred=client.recv();assert pred['ok'],pred

                    for action in np.asarray(pred['raw_h10'],np.float32)[:min(5,config()['replay_max_steps']-steps)]:

                        obs,r,done,_=env.step(action.tolist());rw.append(float(r));steps+=1;success=bool(env.check_success());trace.append(env.get_sim_state().copy())

                        if done or success:break

                    chunk+=1

                trace_path=EXP/f'data/performance2/diagnostics/mixed_physical015/case{a.case_id:03d}_{name}.npz';trace_path.parent.mkdir(parents=True,exist_ok=True)

                assert not trace_path.exists()

                np.savez_compressed(trace_path,initial_full_state=state,requested_h5_actions=actions,normalized_h5_actions=normactions,controller_effective7d_h5=projected,baseline_normalized_h5=base,gradient_scalar=reply['gradients']['scalar'],gradient_divl=reply['gradients']['divl'],baseline_raw_h5=raw_base,rewards=np.asarray(rw,np.float32),states=np.asarray(trace))

                row={'candidate':name,'method_direction':method,'valid':True,'q':q['q'],'return':float(np.dot(rw,.99**np.arange(len(rw)))),

                     'success':success,'observed_suffix_length':len(rw),'censored':not(done or success),

                     'branch_state_error':branch_error,'hidden_state_error':hidden_error,'effective_controller7d_rms':effective_rms,

                     'requested_action6d_rms':float(np.sqrt(np.mean((actions[:,:6]-raw_base[:,:6])**2))),

                     'gripper_fixed':False,'gradient_scope':'Full7d native mapping including gripper; unmatched effective amplitudes excluded from gradient contrasts', 'case_id':a.case_id,'state_source_id':case['state_source_id'],'task_id':a.task_id,'ranking_candidate':name=='baseline' or name.startswith('independent_'),

                     'trace_path':str(trace_path),'trace_sha256':sha(trace_path)}

                result['rows'].append(row);write(path,result);print('PHYSICAL_CANDIDATE',json.dumps(row),flush=True)

            result.update(complete=True,passed=True,valid_states=1)

    except Exception:

        import traceback

        result.update(complete=True,error=traceback.format_exc(),valid_states=0,valid_pairs=0)

    finally:

        env.close();result.update(wall_s=time.monotonic()-began,ended_utc=stamp());write(path,result)





if __name__=='__main__':main()

