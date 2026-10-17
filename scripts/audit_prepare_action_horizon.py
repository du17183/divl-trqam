"""Read code, run real replay/critic/QAM shape trace, freeze outcome-blind protocol."""
import json, numpy as np, jax, jax.numpy as jnp, flax.nnx as nnx
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config
from openpi.training.qam_training import QAMTrainingConfig
from openpi.training.qam_libero_adapter import LiberoQAMBatchBuilder
from openpi.training import config,qam
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.models import model as model_module
from openpi.shared import normalize
from openpi.policies import policy_config
from critic_v2_io import restore_critic
from action_sensitive_v4_common import action_bounds,normalized,raw_action,support_bucket
from horizon_common import R,D,O,REP,SEED,sha,atomic,save_npz,load,report,safe

def main():
 assert not (D/'protocol.json').exists()
 for folder in [D/'candidates',D/'states',D/'episodes',D/'invalid',O,REP]:folder.mkdir(parents=True,exist_ok=True)
 rows=json.loads((R/'data/action_sensitive_replay_v4/state_selection_manifest.json').read_text())['records']
 split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text())
 selected=[]
 for t in range(10):
  taskrows=sorted([r for r in rows if r['task_id']==t],key=lambda r:r['initial_state_id'])
  assert len(taskrows)==30
  # Round-robin over existing phase strata; no branch outcomes enter selection.
  groups={}
  for row in taskrows:groups.setdefault(row['assigned_phase'],[]).append(row)
  rng=np.random.default_rng(SEED+t)
  for phase in groups:rng.shuffle(groups[phase])
  phases=sorted(groups)
  picked=[]
  while len(picked)<10:
   for phase in phases:
    if groups[phase] and len(picked)<10:picked.append(groups[phase].pop())
  selected.extend(sorted(picked,key=lambda r:r['initial_state_id']))
 assert len(selected)==100 and len({(r['task_id'],r['initial_state_id']) for r in selected})==100
 # Real batch from unchanged replay, actual adapter and restored Q head.
 source=load(selected[0]['path']);ix=selected[0]['source_index']
 norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
 enc=encoder_config(R);tokenizer=DIVLTokenizer(enc);mcfg=config.get_config('pi05_libero').model
 qc=QAMTrainingConfig(critic_horizon=10,replan_steps=5,action_dim=7)
 adapter=LiberoQAMBatchBuilder(mcfg,norm,norm,tokenizer,training_config=qc,td_steps=10)
 batch=adapter([(source,ix),(source,ix)])
 critic=restore_critic(build_critic(LiberoTrainingConfig(value_type='scalar'),enc),R/'checkpoints/critic_replay_v2/scalar/step2000')
 @nnx.jit
 def encode(c,obs):return c.encoder(**obs,train=False)
 z=encode(critic,batch.critic['obs'])
 actor=np.zeros((2,10,32),np.float32)
 actor[:,:5,:7]=normalized(source['actions'][ix,:5],*action_bounds())
 low,high=action_bounds();args=[jnp.asarray(low),jnp.asarray(high)]*2
 acts,mask=qam.actions_for_critic(jnp.asarray(actor),*args,replan_steps=5,critic_horizon=10,action_dim=7)
 q1,q2=critic.q_head(z,acts,mask)
 terminal,_=qam.qam_terminal_adjoint(critic,z,jnp.asarray(actor),*args,replan_steps=5,critic_horizon=10,action_dim=7)
 direct=jax.grad(lambda a:jnp.minimum(*critic.q_head(z,a,mask)).sum())(acts)
 term=np.asarray(terminal);direct=np.asarray(direct)
 assert acts.shape==(2,10,7) and mask.shape==(2,10)
 assert np.count_nonzero(term[:,5:])==0 and np.count_nonzero(term[:,:,7:])==0
 assert np.count_nonzero(direct[:,5:])==0
 trace=dict(batch_size=2,state_raw_shape=list(batch.actor_observation.state.shape),raw_proprio_dim=8,
  actor_prediction_shape=list(actor.shape),actor_prediction_axes=['batch','10 predicted timesteps','32 channels, first7 physical'],
  replay_actions_shape=list(batch.critic['actions'].shape),action_mask_shape=list(mask.shape),mask=np.asarray(mask).tolist(),
  critic_encoded_state_shape=list(z.shape),critic_input_action_shape=list(acts.shape),Q1_shape=list(q1.shape),Q2_shape=list(q2.shape),
  QAM_terminal_gradient_shape=list(term.shape),critic_direct_gradient_shape=list(direct.shape),
  terminal_tail_norm=float(np.linalg.norm(term[:,5:])),terminal_padding_norm=float(np.linalg.norm(term[:,:,7:])),
  terminal_prefix_norm=float(np.linalg.norm(term[:,:5,:7])),critic_masked_time_gradient_norm=float(np.linalg.norm(direct[:,5:])),
  prediction_horizon=10,executed_prefix_length=5,critic_temporal_storage_capacity=10,critic_real_action_dimensions=7,
  actor_padded_dimensions=25,actual_training_config='LiberoTrainingConfig(10,5,7); generic QAM default critic_horizon30 is NOT this run',
  td_chunks=10,first_chunk_k=int(source['lengths'][ix]),td_environment_steps=int(source['lengths'][ix:ix+10].sum()),
  obs_shapes={k:{c:list(v.shape) for c,v in val.items()} if isinstance(val,dict) else list(val.shape) for k,val in batch.critic['obs'].items()},
  actor_updated=False,optimizer_created=False)
 # Frozen actual actor VJP trace: solver time is separate from env action time.
 del critic,batch,z
 jax.clear_caches()
 policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),R/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
 raw={'observation/image':source['base'][ix],'observation/wrist_image':source['wrist'][ix],'observation/state':source['state'][ix],'prompt':str(source['prompt'])}
 tr=policy._input_transform(raw)
 ob=model_module.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],tr))
 ob=model_module.preprocess_observation(None,ob,train=False)
 actual=policy._model;actual.eval()
 @nnx.jit
 def backward(a,o,path,times,g):return qam.qam_backward_adjoint(a,o,times,path,g)
 # Saved fixed action/noise path only for dimension/VJP audit, not policy rollout or training.
 times=jnp.linspace(.1,1.,11);path=jnp.tile(jnp.asarray(actor[:1])[None],(11,1,1,1))
 adj=np.asarray(backward(actual,ob,path,times,jnp.asarray(term[:1])))
 trace.update(backward_adjoint_shape=list(adj.shape),flow_integration_intervals=10,
  backward_tail_norm=float(np.linalg.norm(adj[:-1,:,5:])),backward_padding_norm=float(np.linalg.norm(adj[:-1,:,:,7:])),
  backward_prefix_norm=float(np.linalg.norm(adj[:-1,:,:5,:7])),
  backward_trace_note='Actual frozen actor predict_velocity VJP on a fixed replay-endpoint path; shape/dependency audit only, not a sampled QAM training trajectory',
  actor_velocity_loss_time_positions='all10 predicted positions and32channels; no time/channel mask in squared-error sum')
 assert np.isfinite(adj).all()
 atomic(R/'artifacts/action_object_shape_trace.json',trace);atomic(O/'action_object_shape_trace.json',trace)
 del policy,actual,adj; jax.clear_caches()
 codefiles=[R/'src/openpi/training'/n for n in ['qam.py','qam_training.py','qam_libero_adapter.py','divl_libero_adapter.py','divl_lwd_training.py','libero_lwd_training.py']]
 codefiles += [R/'src/openpi/models'/n for n in ['divl_heads.py','divl_losses.py']]
 fact='''## 代码事实（以下行号由本轮源码快照确认）

Q(s,A)输入为[B,10,7]，时间mask仅前5步有效，实际优化对象是**5步真实执行prefix**；10是critic时间容量。Actor预测[B,10,32]；25个padding通道在进入critic前被切掉，并非32D送入Q head。

`qam.py:59-89`转换/归一化/clip/时间padding；`qam.py:119-142`对actor终点[B,10,32]求−min(Q1,Q2)/temperature梯度，仅前5×7坐标直接依赖Q。`divl_heads.py:82-92`以mask attention pooling聚合动作。

`qam.py:252-320`backward adjoint沿10个**flow integration intervals**传播，张量[N+1,B,10,32]；不是10个环境step。冻结flow网络的跨位置Jacobian可以把prefix终端信号传播到其它模型坐标。`qam.py:364-381`velocity matching把所有10×32位置纳入loss；终端padding梯度为0不等于整个backward路径padding梯度恒0。

`libero_lwd_training.py:25-40,58`实际critic为7D、10时间容量、execute5。通用QAM配置默认30，不能冒充本项目实际10。`divl_libero_adapter.py:74-80`训练td_steps10表示10个chunk通常50env steps；首个动作对象仅5step，其余9chunk使用recorded behavior续跑。n-step target不要求动作输入覆盖整个bootstrap窗口。

## 旧diagnostic

`prepare_action_diagnostic.py`生成[8,10,7]候选，mask前k<=5；`run_counterfactual_branches_v2.py`实际执行action[:k]后重新规划；v4同样完整执行5步。旧50-state与v4没有action_object_mismatch。不能把本轮结果写成修复了“旧代码只干预首步”。H-QAM=H5，禁止重复分支。

## 运行时确认

见artifacts/action_object_shape_trace.json：真实LIBERO replay batch经过正式adapter、恢复的Scalar Q、QAM终端求导和真实冻结actor VJP。没有optimizer，没有Actor update。

## 推断与边界

Actor flow的跨时间耦合允许后5预测位置收到间接adjoint，不表示它们被环境执行。旧H5已经与QAM一致，本轮检验短干预长度的影响，并不预设horizon mismatch。原生binary reward下多数非零ΔG可能只是成功完成时间差。
'''
 report('action_object_code_audit.md','Action object code/runtime audit',fact+'\n```json\n'+json.dumps(trace,indent=2)+'\n```')
 # Exactly paired local candidates: reuse label-blind geometry, synthesize symmetric PCA.
 frozenfeatures=load(R/'artifacts/action_sensitive_replay_v4/policy_features.npz')
 lookup={sid:i for i,sid in enumerate(frozenfeatures['state_ids'].tolist())}
 out=[]
 for row in selected:
  assert row['initial_state_id'] in split['tasks'][row['task_name']]['train']
  s=load(row['candidate_path']);base=s['raw'][0].copy();bn=s['normalized'][0].copy()
  # Four local perturbations: two real neighbor interpolations and paired +/- .20 local PCA.
  normplans=np.stack([bn,s['normalized'][1],s['normalized'][2],s['normalized'][4],2*bn-s['normalized'][4]])
  plans=np.zeros((5,10,7),np.float32)
  for j in range(5):plans[j,:5]=raw_action(normplans[j,:5],low,high)
  plans[0]=base
  norms=np.zeros_like(plans);norms[:,:5]=normalized(plans[:,:5],low,high)
  th=json.loads((R/'data/action_sensitive_replay_v4/local_manifold.json').read_text())['thresholds'][str(row['task_id'])]
  old=load(R/'artifacts/replay_action_audit_v3/public_features.npz') if 'old' not in locals() else old
  pool=old['train_actions'][s['support_neighbor_indices'],:5].reshape(10,35)
  specs=[]
  for j,name in enumerate(['behavior','interpolation_low','interpolation_high','local_pca_plus','local_pca_minus']):
   distances={}
   for h in [1,3,5]:
    executed=bn[:5].copy();executed[:h]=norms[j,:h]
    bucket,dist=support_bucket(executed,pool,th)
    distances[str(h)]=dict(bucket=bucket,distance=dist,behavior_rms=float(np.sqrt(np.mean((norms[j,:h]-bn[:h])**2))),
      support_note='Same35D/p50/p90 proxy with unexecuted prefix positions filled by behavior; supplemental, not causal physical support')
   specs.append(dict(index=j,name=name,direction=(norms[j]-bn).tolist(),support=distances))
  sid='ah_'+row['v3_state_id'];p=D/'candidates'/f'{sid}.npz'
  save_npz(p,raw=plans,normalized=norms,behavior=bn,base=frozenfeatures['state_features'][lookup[row['v3_state_id']]])
  out.append({**row,'state_id':sid,'candidate_path':str(p),'candidate_sha256':sha(p),'candidate_specs':specs,
   'v4_state_id':row['state_id'],'feature_index':lookup[row['v3_state_id']]})
 atomic(D/'state_selection_manifest.json',dict(records=out,states=100,selection_before_outcomes=True,unique_task_initial_ids=100))
 frozen=[R/'data/action_sensitive_replay_v4/manifest.json',R/'data/action_sensitive_replay_v4/protocol.json',
  R/'artifacts/v4_representation_probe.json',R/'configs/splits/libero_factorial_v1.json',R/'artifacts/counterfactual_final.parquet',
  R/'checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json']+codefiles
 protocol=dict(version='action_horizon_alignment_20261008',seed=SEED,horizons=[1,3,5],H_QAM_alias=5,states=100,candidates=5,branches=1500,
  gamma=.9999,actor_horizon=10,real_action_dim=7,actor_padded_dim=32,critic_storage_horizon=10,restore_tolerance=1e-5,
  first_intervention='execute firstH of same10step planned candidate; replan immediately atH then cadence5; H5 reproduces source schedule',
  matched_baseline='Each horizon has its OWN unperturbed firstH behavior baseline. H1/H3 do NOT necessarily equal recorded H5 return because earlier replanning is the treatment.',
  continuation_noise='Same source_seed*1000+(source_index+continuation_call) across candidates/horizons; common random numbers by replan ordinal, not forced same real-time noise after changed cadence',
  reproduction='H5 recorded continuation equality plus independent repeat of all3 horizon behavior trajectories, initial flat/RGB/proprio equality. Repeat gate errors <=1e-5, all outcomes retained.',
  intervention_scales='Same exact quantile-normalized per-timestep vectors for H1/H3/H5. Total movement budget grows with horizon by design; not length-normalized.',
  positive_control=dict(folds=5,split='whole(task,initialID), all candidates/horizons grouped',representation='same actual frozen actor visual PCA32 + proprio8 + taskonehot10 + executed35Dprefix padded to5steps; trainfold only scaling',
   architecture=[85,128,64,1],updates=1000,batch_states=64,optimizer='Adam.001',mse_plus_diagnostic_ranking=.1,
   action_label='planned intervention firstH, remaining5-H zero pre-outcome; early success truncation NOT input'),
  recovery=dict(metrics=['proprio_L2','EEF_position_L2','EEF_rotation_radians','mean_object_position_L2','mean_object_rotation_radians'],
   threshold=dict(proprio=.02,eef_position=.005,eef_rotation=.0349066,object_position=.005,object_rotation=.0349066),
   samples='intervention end, after 1/2/3 follow-up5step executions, each later replan, terminal; compare horizon-matched behavior at SAME env elapsed step; no extrapolation after either terminates',
   recovered='all thresholds for two consecutive follow-up replans; unresolved/terminal-censored states retained; fraction and restricted mean(min(replan,3)) reported'),
  decision='ACTION_HORIZON_CONFIRMED only if H5 paired |deltaG| increase and >.01 fraction increase have CIlo>0 versusH1, and H5 OOF rank>=.60 CIlo>.50; otherwise ACTION_HORIZON_REJECTED means bounded hypothesis not supported, not proof of no causal effect. Code/target implementation bugs take precedence.',
  action_object_mismatch=False,actor_updates=False,QAM_updates=False,formal_critic_retrain=False,dev_used=False,final_used=False,
  frozen_input_hashes={str(p):sha(p) for p in frozen},code_hashes={p.name:sha(p) for p in (R/'scripts').glob('*horizon*.py')},
  selection_hash=sha(D/'state_selection_manifest.json'))
 atomic(D/'protocol.json',protocol)
 atomic(O/'audit_complete.json',dict(complete=True,action_object_mismatch=False,H_QAM_alias=5,shape_trace=trace,protocol_hash=sha(D/'protocol.json')))
 print('ACTION_HORIZON_AUDIT_PROTOCOL_COMPLETE',json.dumps(safe(trace)),flush=True)
if __name__=='__main__':main()
