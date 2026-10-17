"""All-data return/recovery/target arithmetic audit and matched fivefold probes."""
import json,time,itertools
import numpy as np,pandas as pd,torch
from torch import nn
from horizon_common import R,D,O,SEED,atomic,save_npz,load,sha,report,safe
from analyze_replay_action_v3 import metrics,clustered
from analyze_action_sensitive_replay_v4 import pca_fit
from action_sensitive_v4_common import action_bounds,normalized

def aggregate(pred,g,ids,tasks,support):
 result={}
 for b in ['all','Near-ID','Near-ID+Mid']:
  rows=[]
  for n,sid in enumerate(ids):
   mask=np.ones(5,bool) if b=='all' else support[n]=='Near-ID' if b=='Near-ID' else support[n]!='Far-OOD'
   m=metrics(pred[n,mask],g[n,mask]);m.update(state_id=sid,task_id=int(tasks[n]));rows.append(m)
  result[b]=dict(rank=clustered(rows,'rank'),spearman=clustered(rows,'spearman'),states=rows,effective_pairs=sum(v['pairs'] for v in rows))
 return result
def quaternion_angle(a,b):
 dot=np.clip(np.abs(np.sum(a*b,axis=-1)),0,1);return 2*np.arccos(dot)
def deviation(e,b,t):
 if t>=len(e['physical_eef_pos']) or t>=len(b['physical_eef_pos']):return None
 # Actor proprio exactly8D, not quaternion9D; both raw policy frames reconstructed analytically.
 def proprio(ep):
  q=ep['physical_eef_quat'][t];w=np.clip(q[3],-1,1);den=np.sqrt(1-w*w)
  aa=np.zeros(3) if den<1e-8 else q[:3]*(2*np.arccos(w)/den)
  return np.r_[ep['physical_eef_pos'][t],aa,ep['physical_proprio'][t][-2:]]
 n=e['object_positions'].shape[1]
 return dict(proprio_L2=float(np.linalg.norm(proprio(e)-proprio(b))),
  EEF_position_L2=float(np.linalg.norm(e['physical_eef_pos'][t]-b['physical_eef_pos'][t])),
  EEF_rotation_radians=float(quaternion_angle(e['physical_eef_quat'][t],b['physical_eef_quat'][t])),
  mean_object_position_L2=float(np.linalg.norm(e['object_positions'][t]-b['object_positions'][t],axis=-1).mean()) if n else 0.,
  mean_object_rotation_radians=float(quaternion_angle(e['object_quaternions'][t],b['object_quaternions'][t]).mean()) if n else 0.)

def main():
 torch.set_num_threads(2);started=time.monotonic();m=json.loads((D/'manifest.json').read_text());p=json.loads((D/'protocol.json').read_text())
 assert m['states']==100 and m['outcomes']==1500 and m['protocol_hash']==sha(D/'protocol.json')
 select=json.loads((D/'state_selection_manifest.json').read_text())['records'];lookup={r['state_id']:r for r in select};ids=sorted(lookup)
 tasks=np.array([lookup[s]['task_id'] for s in ids]);initial=np.array([lookup[s]['initial_state_id'] for s in ids])
 grouped={}
 for row in m['records']:grouped.setdefault((row['state_id'],row['horizon']),[]).append(row)
 zf=load(R/'artifacts/action_sensitive_replay_v4/policy_features.npz');z=np.stack([zf['state_features'][lookup[s]['feature_index']] for s in ids])
 proprio=[]
 for sid in ids:
  st=load(R/'data/action_sensitive_replay_v4/states'/f'{lookup[sid]["v4_state_id"]}.npz');proprio.append(st['proprio'])
 proprio=np.stack(proprio);returns={};actions={};support={};branches=[];recover=[];targetrows=[];audit_errors=[]
 low,high=action_bounds()
 for h in [1,3,5]:
  gg=[];aa=[];ss=[]
  for sid in ids:
   rows=sorted(grouped[(sid,h)],key=lambda r:r['candidate_index']);assert len(rows)==5
   base=load(rows[0]['path']);g=np.array([r['future_return'] for r in rows]);gg.append(g);ss.append([r['bucket'] for r in rows]);an=[]
   for row in rows:
    assert sha(row['path'])==row['sha256'];e=base if row['candidate_index']==0 else load(row['path'])
    expected=np.dot(.9999**np.arange(len(e['env_step_rewards'])),e['env_step_rewards'])
    assert abs(expected-row['future_return'])<=1e-6
    offset=0;rewardmax=0.;discountmax=0.
    for ix,k in enumerate(e['lengths']):
     rw=float(np.dot(.9999**np.arange(k),e['env_step_rewards'][offset:offset+k]));rewardmax=max(rewardmax,abs(rw-float(e['rewards'][ix])))
     dis=0. if e['terminal'][ix] else .9999**int(k);discountmax=max(discountmax,abs(dis-float(e['discounts'][ix])))
     assert np.array_equal(e['actions'][ix,e['action_mask'][ix]],e['raw_env_actions'][offset:offset+k]);offset+=int(k)
    end=min(10,len(e['rewards']));reward=0.;discount=1.;actualend=0
    for j in range(end):
     reward+=discount*float(e['rewards'][j]);discount*=float(e['discounts'][j]);actualend=j+1
     if discount==0:break
    ktd=int(e['lengths'][:actualend].sum());nativewindow=float(np.dot(.9999**np.arange(ktd),e['env_step_rewards'][:ktd]))
    err=abs(nativewindow-reward);assert err<=1e-6 and rewardmax<=1e-6 and discountmax<=1e-7
    Gtail=float(e['mc_returns'][actualend]) if actualend<len(e['mc_returns']) else 0.
    decomposition=abs(reward+discount*Gtail-float(e['mc_returns'][0]));assert decomposition<=1e-6
    targetrows.append(dict(episode_id=row['episode_id'],state_id=sid,horizon=h,candidate_index=row['candidate_index'],path=row['path'],
      Target_A=float(e['mc_returns'][0]),Target_B=float(e['rewards'][0]),first_k=int(e['lengths'][0]),td_chunks=actualend,td_environment_steps=ktd,
      td_window_reward=reward,td_discount=discount,bootstrap_state_index=actualend,native_window_error=err,MC_decomposition_error=decomposition,
      terminal_window=discount==0,terminal_reward_included=bool(e['success']),reward_scaling=1.,gamma=.9999))
    audit_errors.append(dict(chunk_reward_error=rewardmax,discount_error=discountmax,MC_error=abs(expected-row['future_return']),window_error=err,decomposition_error=decomposition))
    plan=normalized(e['candidate_plan_raw'][:5],low,high);plan[h:]=0;an.append(plan.reshape(35))
    branches.append({**row,'delta_G':float(g[row['candidate_index']]-g[0]),'absolute_delta_G':float(abs(g[row['candidate_index']]-g[0])),
      'success_flip':bool(row['success']!=rows[0]['success'])})
    if row['candidate_index']:
     for label,t in [('intervention_end',h),('replan1',h+5),('replan2',h+10),('replan3',h+15),('terminal',int(e['lengths'].sum()))]:
      dev=deviation(e,base,t)
      recover.append(dict(state_id=sid,task_id=row['task_id'],horizon=h,candidate_index=row['candidate_index'],label=label,env_step=t,
        valid_same_time=dev is not None,branch_terminal_step=int(e['lengths'].sum()),baseline_terminal_step=int(base['lengths'].sum()),**(dev or {})))
     rr=None;observed=0
     threshold=p['recovery']['threshold']
     def close(d):return d is not None and d['proprio_L2']<=threshold['proprio'] and d['EEF_position_L2']<=threshold['eef_position'] and d['EEF_rotation_radians']<=threshold['eef_rotation'] and d['mean_object_position_L2']<=threshold['object_position'] and d['mean_object_rotation_radians']<=threshold['object_rotation']
     for n in range(1,1+min((len(e['physical_eef_pos'])-1-h)//5,(len(base['physical_eef_pos'])-1-h)//5)):
      observed=n
      if close(deviation(e,base,h+5*n)) and close(deviation(e,base,h+5*(n+1))):rr=n;break
     recover.append(dict(state_id=sid,task_id=row['task_id'],horizon=h,candidate_index=row['candidate_index'],label='recovery_time',
      recovered_replans=rr,observed_replans=observed,restricted_replans_3=min(rr,3) if rr is not None else 3,
      censored=rr is None,valid_same_time=True))
   aa.append(an)
  returns[h]=np.stack(gg);actions[h]=np.stack(aa);support[h]=np.asarray(ss)
 pd.DataFrame(branches).to_parquet(R/'artifacts/action_horizon_branches.parquet',index=False)
 pd.DataFrame(targetrows).to_parquet(O/'target_requests.parquet',index=False)
 pd.DataFrame(recover).to_parquet(O/'recovery_samples.parquet',index=False)
 summaries={};allpair=[]
 for h in [1,3,5]:
  br=[r for r in branches if r['horizon']==h and r['candidate_index']>0];ag=np.array([r['absolute_delta_G'] for r in br])
  pairvals=np.array([abs(v[i]-v[j]) for v in returns[h] for i,j in itertools.combinations(range(5),2)])
  st=[]
  for n,sid in enumerate(ids):
   v=returns[h][n];ds=np.abs(v[1:]-v[0]);row=dict(state_id=sid,task_id=int(tasks[n]),mean_abs_delta=float(ds.mean()),
     probability_delta_gt_01=float(np.mean(ds>.01)),success_flip_rate=float(np.mean([r['success_flip'] for r in br if r['state_id']==sid])))
   st.append(row)
  summaries[h]=dict(branches=500,perturbed_branches=400,states=100,H_QAM_alias=h==5,
   mean_absolute_delta_G=float(ag.mean()),median_absolute_delta_G=float(np.median(ag)),
   probabilities={str(t):float(np.mean(ag>t)) for t in [1e-4,1e-3,1e-2]},success_flip_rate=float(np.mean([r['success_flip'] for r in br])),
   mean_abs_delta_CI=clustered(st,'mean_abs_delta'),delta_gt_01_CI=clustered(st,'probability_delta_gt_01'),
   all_candidate_pair_probabilities={str(t):float(np.mean(pairvals>t)) for t in [1e-6,1e-4,1e-3,1e-2]},
   source_H5_gate_max=max(a['source_return_error'] for a in m['restore_audits'] if a['horizon']==5) if h==5 else None,
   repeat_restore_max=max(a['repeat_return_error'] for a in m['restore_audits'] if a['horizon']==h),initial_state_error_max=0.,
   support_counts={b:int(np.sum(support[h]==b)) for b in ['Near-ID','Mid','Far-OOD']},state_metrics=st)
 paired={}
 for l,r in [(3,1),(5,1),(5,3)]:
  diffs=[]
  for a,b in zip(summaries[l]['state_metrics'],summaries[r]['state_metrics'],strict=True):
   assert a['state_id']==b['state_id'];diffs.append(dict(state_id=a['state_id'],task_id=a['task_id'],mean_abs_delta=a['mean_abs_delta']-b['mean_abs_delta'],probability_delta_gt_01=a['probability_delta_gt_01']-b['probability_delta_gt_01'],success_flip_rate=a['success_flip_rate']-b['success_flip_rate']))
  paired[f'H{l}_minus_H{r}']={k:clustered(diffs,k) for k in ['mean_abs_delta','probability_delta_gt_01','success_flip_rate']}
 sensitivity=dict(horizons=summaries,paired=paired,H_QAM_alias='H5, no duplicated branches',sampling_unit='whole state,task-stratified bootstrap',candidate_direction_seed_identical=True)
 atomic(R/'artifacts/action_horizon_metrics.json',safe(sensitivity))
 curves={};recoverytimes={}
 for h in [1,3,5]:
  curves[h]={}
  for label in ['intervention_end','replan1','replan2','replan3','terminal']:
   rows=[r for r in recover if r['horizon']==h and r['label']==label];valid=[r for r in rows if r['valid_same_time']]
   out=dict(available_branches=len(valid),censored_branches=len(rows)-len(valid))
   for key in ['proprio_L2','EEF_position_L2','EEF_rotation_radians','mean_object_position_L2','mean_object_rotation_radians']:
    state=[]
    for sid in ids:
     vv=[v[key] for v in valid if v['state_id']==sid]
     if vv:state.append(dict(state_id=sid,task_id=lookup[sid]['task_id'],value=float(np.mean(vv))))
    out[key]=clustered(state,'value')
   curves[h][label]=out
  rt=[r for r in recover if r['horizon']==h and r['label']=='recovery_time'];valid=[r for r in rt if r['recovered_replans'] is not None]
  recoverytimes[h]=dict(branches=400,ever_recovered_fraction=len(valid)/400,recovered_by_replan3_fraction=sum(r['recovered_replans'] is not None and r['recovered_replans']<=3 for r in rt)/400,
    mean_replans_among_recovered=float(np.mean([r['recovered_replans'] for r in valid])) if valid else None,
    restricted_mean_replans3=float(np.mean([r['restricted_replans_3'] for r in rt])),unresolved_or_censored=400-len(valid),
    mean_note='Recovered-only mean excludes unresolved; restricted mean caps all400branches at3, with unresolved counted3. Not an unrestricted recovery-time estimate.')
 recoveryobj=dict(curves=curves,recovery_times=recoverytimes,thresholds=p['recovery'],visual_embedding_curve='Not computed; physical/current proprio and object poses recorded at every native step.')
 atomic(R/'artifacts/recovery_curve_metrics.json',safe(recoveryobj))
 # Small, same-capacity/state split positive control for each intervention horizon.
 folds=np.zeros(100,int)
 for t in range(10):
  ix=np.flatnonzero(tasks==t);ix=ix[np.argsort(initial[ix])];folds[ix]=np.arange(len(ix))%5
 pred={h:np.zeros((100,5),np.float32) for h in [1,3,5]};foldrecords={h:[] for h in pred}
 ii,jj=map(torch.tensor,np.triu_indices(5,1))
 for f in range(5):
  tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);zt,ze=pca_fit(z,tr,te)
  pm=proprio[tr].mean(0);ps=np.maximum(proprio[tr].std(0),1e-3);pt=(proprio[tr]-pm)/ps;pe=(proprio[te]-pm)/ps
  # Shared per-fold scaler pooled equally across all three action horizons, without heldout/outcomes.
  def inp(ix,zz,pp,h):
   s=np.concatenate([zz,pp,np.eye(10)[tasks[ix]]],-1).astype(np.float32)
   return np.concatenate([np.repeat(s[:,None],5,axis=1),actions[h][ix]],-1)
  pool=np.concatenate([inp(tr,zt,pt,h).reshape(-1,85) for h in pred]);mean=pool.mean(0);std=np.maximum(pool.std(0),.05)
  for h in pred:
   x=torch.tensor((inp(tr,zt,pt,h)-mean)/std);xt=torch.tensor((inp(te,ze,pe,h)-mean)/std);truth=torch.tensor(returns[h][tr],dtype=torch.float32)
   torch.manual_seed(SEED+f);rng=np.random.default_rng(SEED+f);net=nn.Sequential(nn.Linear(85,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1));opt=torch.optim.Adam(net.parameters(),lr=.001)
   for step in range(1000):
    sample=torch.tensor(rng.integers(len(tr),size=64));q=net(x[sample]).squeeze(-1);y=truth[sample];delta=y[:,ii]-y[:,jj];valid=abs(delta)>1e-6
    rank=nn.functional.softplus(-torch.sign(delta)*(q[:,ii]-q[:,jj])/.1)[valid].mean() if valid.any() else q.sum()*0
    loss=((q-y)**2).mean()+.1*rank;opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():pred[h][te]=net(xt).squeeze(-1).numpy()
   out=aggregate(pred[h][te],returns[h][te],[ids[n] for n in te],tasks[te],support[h][te]);foldrecords[h].append(dict(fold=f,train_states=len(tr),heldout_states=len(te),parameter_count=sum(v.numel() for v in net.parameters()),metrics=out,train_loss=float(loss)))
   print('HORIZON_PROBE_FOLD',h,f,'ranking',out['all']['rank'],flush=True)
 probes={h:dict(metrics=aggregate(pred[h],returns[h],ids,tasks,support[h]),folds=foldrecords[h]) for h in pred}
 pairedrank={}
 for l,r in [(3,1),(5,1),(5,3)]:
  diff=[]
  for a,b in zip(probes[l]['metrics']['all']['states'],probes[r]['metrics']['all']['states'],strict=True):
   diff.append(dict(state_id=a['state_id'],task_id=a['task_id'],rank=a['rank']-b['rank'] if np.isfinite(a['rank']) and np.isfinite(b['rank']) else np.nan,spearman=a['spearman']-b['spearman'] if np.isfinite(a['spearman']) and np.isfinite(b['spearman']) else np.nan))
  pairedrank[f'H{l}_minus_H{r}']={key:clustered(diff,key) for key in ['rank','spearman']}
 atomic(R/'artifacts/horizon_positive_control.json',safe(dict(horizons=probes,paired=pairedrank,protocol=p['positive_control'],formal_training=False,actor_updated=False)))
 save_npz(O/'probe_predictions.npz',H1=pred[1],H3=pred[3],H5=pred[5],G1=returns[1],G3=returns[3],G5=returns[5],folds=folds,state_ids=np.asarray(ids))
 targetaudit=dict(arithmetic_errors={k:max(v[k] for v in audit_errors) for k in audit_errors[0]},reward_scaling=1.,gamma=.9999,
  action_object='first5 executed real actions',td_steps='10CHUNKS (not10env actions); normallyH+45 env steps in this diagnostic',
  Target_A='full discounted native reward future from the original snapshot',Target_B='first actual executed chunk discounted native reward sum',
  Target_C='actual TD10chunk reward+product(stored discounts)*EMA target V at boundary; computed next for both frozenv2 value versions',
  MC_decomposition='A=TDwindow_reward+TDdiscount*MC_future_at_bootstrap; numerical equality audited; V != recorded MC is intentional bootstrap approximation, not by itself a bug',
  terminal='first success/native done/native time limit; terminal window discount0, include successful terminal reward',implementation_bug=False,bootstrap_values_pending=True)
 atomic(O/'target_arithmetic.json',safe(targetaudit))
 report('action_horizon_branch_report.md','Paired horizon branches',json.dumps(safe(dict(states=100,branches=1500,behavior_repeat_rollouts=300,restore_audits=m['restore_audits'],protocol_hash=m['protocol_hash'])),indent=2))
 report('action_horizon_return_sensitivity.md','Future return sensitivity',json.dumps(safe(sensitivity),indent=2)+'\n\nEach H has its own behavior baseline; earlier replan changes behavior even with zero perturbation. Same snapshot, candidate vectors and continuation random numbers by call ordinal. Per-step amplitude fixed; total budget grows with H. All candidates/flat outcomes retained.')
 report('policy_recovery_curve.md','Policy recovery curves',json.dumps(safe(recoveryobj),indent=2)+'\n\nSame elapsed env-step comparison; no extrapolation beyond either terminal. EEF translation/object distance in meters; rotations in radians; proprio is actual8D policy representation. Censored branches remain in denominators.')
 report('action_horizon_positive_control.md','Matched horizon positive controls',json.dumps(safe(dict(horizons=probes,paired=pairedrank,protocol=p['positive_control'])),indent=2)+'\n\n100whole initialID states,5fold, all horizon/candidate branches grouped; actual actor visual features, trainfold PCA/scalers; identical capacity19,329/optimizer/budget/seeds. No post-outcome mask, success, remaining horizon or oracle features are model input. No formalcritic or actor is trained.')
 atomic(O/'analysis_complete.json',dict(complete=True,elapsed_seconds=time.monotonic()-started,target_bootstrap_pending=True))
 print('HORIZON_CPU_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
