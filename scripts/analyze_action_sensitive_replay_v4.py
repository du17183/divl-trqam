"""Full retained-outcome audit and fair A/B/C/GT whole-state CV probes."""
import itertools,json,numpy as np,pandas as pd,torch,time
from torch import nn
from action_sensitive_v4_common import R,D,O,REP,SEED,sha,atomic
from analyze_replay_action_v3 import metrics,clustered,safe,stats
from evaluate_action_sensitive_replay_v2 import first_rows

def aggregate(pred,g,support,ids,tasks):
 result={}
 for bucket in ['all','Near-ID','Mid','Far-OOD']:
  rows=[]
  for i,sid in enumerate(ids):
   idx=np.arange(7) if bucket=='all' else np.flatnonzero(support[i]==bucket)
   row=metrics(pred[i,idx],g[i,idx]);row.update(state_id=sid,task_id=int(tasks[i]));rows.append(row)
  result[bucket]=dict(rank=clustered(rows,'rank'),spearman=clustered(rows,'spearman'),effective_pairs=sum(x['pairs'] for x in rows),states=rows)
 return result

def paired(left,right):
 result={}
 for bucket in ['all','Near-ID','Mid','Far-OOD']:
  rows=[]
  for a,b in zip(left[bucket]['states'],right[bucket]['states'],strict=True):
   assert (a['state_id'],a['task_id'],a['pairs'],a['candidates'])==(b['state_id'],b['task_id'],b['pairs'],b['candidates'])
   row=dict(state_id=a['state_id'],task_id=a['task_id'])
   for key in ['rank','spearman']:
    row[key]=float(a[key]-b[key]) if np.isfinite(a[key]) and np.isfinite(b[key]) else np.nan
   rows.append(row)
  result[bucket]={key:clustered(rows,key) for key in ['rank','spearman']}
 return result

def pca_fit(z,tr,te):
 center=z[tr].mean(0);_,_,v=np.linalg.svd(z[tr]-center,full_matrices=False);basis=v[:min(32,len(tr)-1,z.shape[1])].T
 a=(z[tr]-center)@basis;b=(z[te]-center)@basis;std=np.maximum(a.std(0),1e-3);a/=std;b/=std
 return np.pad(a,((0,0),(0,32-a.shape[1]))),np.pad(b,((0,0),(0,32-b.shape[1])))

def main():
 torch.set_num_threads(2);started=time.monotonic()
 protocol=json.loads((D/'protocol.json').read_text());manifest=json.loads((D/'manifest.json').read_text())
 assert manifest['config_hash']==sha(D/'protocol.json')
 split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text());groups={}
 for row in manifest['records']:groups.setdefault(row['state_id'],[]).append(row)
 frozen=np.load(O/'policy_features.npz',allow_pickle=False);lookup={sid:i for i,sid in enumerate(frozen['state_ids'].tolist())}
 ids=sorted(groups);g=[];a=[];z=[];p=[];gt=[];support=[];tasks=[];initial=[];states=[];branches=[];maxerror=0;truncations=0
 for sid in ids:
  rows=sorted(groups[sid],key=lambda x:x['candidate_index']);assert len(rows)==7
  with np.load(D/'states'/f'{sid}.npz',allow_pickle=False) as s:
   norm=s['candidate_normalized'];raw=s['candidate_raw'];assert np.count_nonzero(s['candidate_padded32'][:,:,7:])==0
   proprio=s['proprio'];base=s['base'];wrist=s['wrist'];gt.append(s['gt_diagnostic'])
  v=lookup[sid[3:]];z.append(frozen['state_features'][v]);p.append(proprio);a.append(norm[:,:5].reshape(7,35))
  returns=np.array([r['future_return'] for r in rows],np.float32);g.append(returns);buckets=[r['support_bucket'] for r in rows];support.append(buckets)
  tasks.append(rows[0]['task_id']);initial.append(rows[0]['initial_state_id'])
  for j,row in enumerate(rows):
   assert row['initial_state_id'] in split['tasks'][row['task_name']]['train'];assert sha(row['path'])==row['sha256']
   first=first_rows(row['path'],['base','wrist','state','actions','action_mask','env_step_rewards'])
   assert np.array_equal(first['base'],base) and np.array_equal(first['wrist'],wrist) and np.array_equal(first['state'],proprio)
   mask=first['action_mask'];assert np.array_equal(first['actions'][mask],raw[j][mask]);truncations+=int(mask.sum()<5)
   with np.load(row['path'],allow_pickle=False) as e:
    rewards=e['env_step_rewards'];computed=float(np.dot(.9999**np.arange(len(rewards)),rewards))
   assert abs(computed-row['future_return'])<1e-6
   if j==0:maxerror=max(maxerror,row['baseline_reproduction_error'])
   branches.append({**row,'delta_from_behavior':float(returns[j]-returns[0]),'absolute_delta_from_behavior':float(abs(returns[j]-returns[0]))})
  counts={str(t):sum(abs(float(returns[i]-returns[j]))>t for i,j in itertools.combinations(range(7),2)) for t in [1e-6,1e-4,1e-3,1e-2]}
  near=np.flatnonzero(np.asarray(buckets)=='Near-ID');pairs=list(itertools.combinations(near,2));nearcounts={str(t):sum(abs(float(returns[i]-returns[j]))>t for i,j in pairs) for t in [1e-6,1e-4,1e-3,1e-2]}
  successes=[r['success'] for r in rows]
  states.append(dict(state_id=sid,task_id=rows[0]['task_id'],initial_state_id=rows[0]['initial_state_id'],phase=rows[0]['phase'],
    return_range=float(np.ptp(returns)),effective_pairs=counts,near_actions=len(near),near_pairs=len(pairs),near_effective_pairs=nearcounts,mixed_outcomes=any(successes) and not all(successes)))
 g=np.stack(g);a=np.stack(a);z=np.stack(z);p=np.stack(p);gt=np.stack(gt);support=np.asarray(support);tasks=np.array(tasks);initial=np.array(initial)
 pd.DataFrame(branches).to_parquet(R/'artifacts/v4_branch_metrics.parquet',index=False)
 nearpairs=sum(s['near_pairs'] for s in states)
 summary=dict(states=len(ids),outcomes=len(branches),attempted_states=manifest['attempted_states'],invalid_states=manifest['invalid_states'],
  task_counts={str(t):int((tasks==t).sum()) for t in range(10)},support_counts={b:int((support==b).sum()) for b in ['Near-ID','Mid','Far-OOD']},
  near_candidate_ratio=float(np.mean(support=='Near-ID')),near_effective_states=sum(s['near_effective_pairs']['1e-06']>0 for s in states),
  near_actions_per_state=float(np.mean([s['near_actions'] for s in states])),near_pairs=nearpairs,
  near_supervision={str(t):dict(effective_pairs=sum(s['near_effective_pairs'][str(t)] for s in states),
    fraction_of_all_near_pairs=sum(s['near_effective_pairs'][str(t)] for s in states)/max(nearpairs,1),
    informative_states=sum(s['near_effective_pairs'][str(t)]>0 for s in states),mean_effective_pairs_per_state=float(np.mean([s['near_effective_pairs'][str(t)] for s in states]))) for t in [1e-6,1e-4,1e-3,1e-2]},
  informative={str(t):dict(states=sum(s['return_range']>t for s in states),mean_effective_pairs=float(np.mean([s['effective_pairs'][str(t)] for s in states]))) for t in [1e-6,1e-4,1e-3,1e-2]},
  mixed_success_failure_states=sum(s['mixed_outcomes'] for s in states),baseline_max_error=maxerror,post_outcome_prefix_truncations=truncations,
  all_outcomes_retained=True,RGB_proprio_identical=True,GT_diagnostic_only=True)
 types={}
 for kind in ['behavior','interpolation','local_noise','local_pca','mid_pca']:
  rr=[r for r in branches if r['candidate_kind']==kind];near=[r for r in rr if r['support_bucket']=='Near-ID']
  types[kind]=dict(outcomes=len(rr),near_outcomes=len(near),action_distance=stats([r['behavior_rms'] for r in rr]),
   delta_G=stats([r['absolute_delta_from_behavior'] for r in rr]),near_delta_G=stats([r['absolute_delta_from_behavior'] for r in near]),
   near_different_from_behavior={str(t):sum(r['absolute_delta_from_behavior']>t for r in near) for t in [1e-6,1e-4,1e-3,1e-2]},
   success_ratio=float(np.mean([r['success'] for r in rr])),support_counts={b:sum(r['support_bucket']==b for r in rr) for b in ['Near-ID','Mid','Far-OOD']})
 audit=dict(summary=summary,candidate_types=types,states=states)
 atomic(O/'supervision.json',safe(audit));np.savez_compressed(O/'probe_data.npz',features=z,proprio=p,gt=gt,actions=a,returns=g,support=support,tasks=tasks,initial=initial,state_ids=np.asarray(ids))
 for name,title,obj in [('v4_near_id_coverage.md','Near-ID coverage and fixed thresholds',summary),('v4_action_supervision.md','All outcomes retained',audit),('v4_branch_collection.md','Real same-state branches',summary)]:
  (REP/name).write_text('# '+title+'\n\n'+json.dumps(safe(obj),indent=2)+'\n\nNear-ID remains an empirical public-feature neighborhood proxy, not established physical support. No flat-return state or failed branch is removed. Informative Near-ID state means at least one strict within-Near-ID pair, |deltaG|>1e-6. Timing-only effects are reported separately by delta thresholds.\n')
 print('V4_AUDIT_COMPLETE',json.dumps(safe(summary)),flush=True)
 folds=np.zeros(len(ids),int)
 for t in range(10):
  for rank,init in enumerate(sorted(set(initial[tasks==t]))):folds[(tasks==t)&(initial==init)]=rank%5
 predictions={name:np.zeros_like(g) for name in ['A','B','C','D']};fold_records={name:[] for name in predictions}
 ii,jj=np.triu_indices(7,1);ii=torch.tensor(ii);jj=torch.tensor(jj)
 for fold in range(5):
  tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);ztr,zte=pca_fit(z,tr,te);gttr,gtte=pca_fit(gt,tr,te)
  pm=p[tr].mean(0);ps=np.maximum(p[tr].std(0),1e-3);ptr=(p[tr]-pm)/ps;pte=(p[te]-pm)/ps
  for name in ['A','B','C','D']:
   embedding_tr,embedding_te=(gttr,gtte) if name=='D' else (np.zeros_like(ztr),np.zeros_like(zte)) if name=='C' else (ztr,zte)
   proptr,propte=(np.zeros_like(ptr),np.zeros_like(pte)) if name=='A' else (ptr,pte)
   def inputs(indices,embed,prop):
    static=np.concatenate([embed,prop,np.eye(10)[tasks[indices]]],1).astype(np.float32)
    return np.concatenate([np.repeat(static[:,None],7,axis=1),a[indices]],-1)
   x=inputs(tr,embedding_tr,proptr);xt=inputs(te,embedding_te,propte)
   mean=x.reshape(-1,x.shape[-1]).mean(0);std=np.maximum(x.reshape(-1,x.shape[-1]).std(0),.05)
   x=torch.tensor((x-mean)/std);xt=torch.tensor((xt-mean)/std);truth=torch.tensor(g[tr])
   torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold)
   net=nn.Sequential(nn.Linear(85,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1));assert x.shape[-1]==85
   optimizer=torch.optim.Adam(net.parameters(),lr=.001)
   for step in range(1000):
    sampled=torch.tensor(rng.integers(len(tr),size=64));q=net(x[sampled]).squeeze(-1);y=truth[sampled];delta=y[:,ii]-y[:,jj];valid=abs(delta)>1e-6
    rank=nn.functional.softplus(-torch.sign(delta)*(q[:,ii]-q[:,jj])/.1)[valid].mean() if valid.any() else q.sum()*0
    loss=((q-y)**2).mean()+.1*rank;optimizer.zero_grad();loss.backward();optimizer.step()
   with torch.no_grad():predictions[name][te]=net(xt).squeeze(-1).numpy()
   agg=aggregate(predictions[name][te],g[te],support[te],[ids[i] for i in te],tasks[te])
   record=dict(fold=fold,train_states=len(tr),heldout_states=len(te),metrics={b:{k:v for k,v in val.items() if k!='states'} for b,val in agg.items()},
    identical_parameter_count=sum(v.numel() for v in net.parameters()),capacity=[85,128,64,1],optimizer='Adam.001',steps=1000,final_train_loss=float(loss))
   fold_records[name].append(record);print('V4_PROBE_FOLD',name,fold,json.dumps(safe(record)),flush=True)
 probes={}
 for name in predictions:
  agg=aggregate(predictions[name],g,support,ids,tasks)
  rank=agg['all']['rank'];near=agg['Near-ID']['rank'];rho=agg['all']['spearman']
  passed=bool(rank['mean'] is not None and rank['mean']>=.60 and rank['ci95'][0]>.50 and near['mean'] is not None and near['mean']>=.65 and near['states']>=100 and rho['ci95'] is not None and rho['ci95'][0]>0)
  taskmacro=np.mean([np.nanmean([s['rank'] for s in agg['all']['states'] if s['task_id']==t]) for t in range(10)])
  probes[name]=dict(metrics=agg,folds=fold_records[name],fold_macro_ranking=np.mean([v['metrics']['all']['rank']['mean'] for v in fold_records[name]]),task_macro_ranking=taskmacro,passed=passed,formal_eligible=name!='D')
 pairs={f'{l}_minus_{r}':paired(probes[l]['metrics'],probes[r]['metrics']) for l,r in [('B','A'),('C','B'),('D','B')]}
 primary=probes['B'];gd=pairs['D_minus_B']['all']['rank'];gtmean=probes['D']['metrics']['all']['rank']['mean']
 representation=(not primary['passed'] and gd['mean'] is not None and gd['mean']>=.10 and gd['ci95'][0]>0 and gtmean>=.65)
 decision='READY_FOR_FORMAL_V4_CRITICS' if primary['passed'] else 'STOP_REPRESENTATION' if representation else 'STOP_DATA_REDESIGN'
 result=dict(probes=probes,paired=pairs,primary_probe='B',decision=decision,gate_passed=primary['passed'],protocol=protocol['positive_control'],
  no_oracle_formal_inputs=True,actor_updated=False,elapsed_seconds=time.monotonic()-started,
  GT_limitation='Current GT is a fixed geometry/contact surrogate with train-only PCA32; its failure cannot prove theoretical unlearnability or that all hidden simulator state is observable.')
 atomic(R/'artifacts/v4_representation_probe.json',safe(result));atomic(R/'artifacts/v4_positive_control.json',safe(dict(primary_probe='B',passed=primary['passed'],decision=decision,metrics=primary['metrics'],folds=primary['folds'],gate=protocol['positive_control']['gate'])))
 np.savez_compressed(O/'probe_predictions.npz',**predictions,returns=g,folds=folds,state_ids=np.asarray(ids))
 compact={name:{k:v for k,v in val.items() if k!='metrics'}|{'metrics':{b:{k:v for k,v in metric.items() if k!='states'} for b,metric in val['metrics'].items()}} for name,val in probes.items()}
 body=json.dumps(safe(dict(decision=decision,probes=compact,paired=pairs,protocol=protocol['positive_control'])),indent=2)
 for name,title in [('v4_representation_probe.md','A/B/C/GT fair representation probes'),('v4_positive_control.md','Preregistered primary B gate')]:
  (REP/name).write_text('# '+title+'\n\n'+body+'\n\nWhole(task,initialID) fivefold split, all seven actions together. All models85/128/64/1, same initialization/order/optimizer/1000 updates. A uses actual frozen actor visual embeddings; B adds explicit8Dproprio; C removes vision; D is diagnostic-only current GT. Zeros pad unused slots so parameter counts match. PCA/scalers fit within trainingfold only. Primary B fixed before outcomes; no best-probe selection permits formal training.\n')
 atomic(O/'analysis_complete.json',dict(complete=True,decision=decision,gate_passed=primary['passed'],actor_updated=False))
 print('V4_PROBES_COMPLETE',decision,json.dumps(safe({n:{b:v['rank'] for b,v in p['metrics'].items()} for n,p in probes.items()})),flush=True)

if __name__=='__main__':main()
