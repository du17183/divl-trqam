"""Outcome-blind local candidates; freeze before any new branch execution."""
import json,numpy as np,collections,pathlib
from action_sensitive_v4_common import R,D,O,REP,SEED,sha,atomic,action_bounds,normalized,raw_action,support_bucket

def main():
 assert not (D/'protocol.json').exists()
 f=np.load(O/'policy_features.npz',allow_pickle=False);meta=json.loads(str(f['pool_metadata']))
 tasks=np.array([r['task_id'] for r in meta]);eligible=f['pool_masks'][:,:5].all(1)
 rows=json.loads((R/'data/action_sensitive_replay_v3/state_selection_manifest.json').read_text())['records']
 old=np.load(R/'artifacts/replay_action_audit_v3/public_features.npz',allow_pickle=False)
 sf=np.load(R/'artifacts/action_sensitive_replay_v3/v3_frozen_features.npz',allow_pickle=False)
 oldmeta=json.loads(str(old['train_metadata']));oldtask=np.array([r['task_id'] for r in oldmeta])
 thresholds=json.loads((R/'artifacts/replay_action_audit_v3/ood_analysis.json').read_text())['calibration']
 actor_lookup={sid:i for i,sid in enumerate(f['state_ids'].tolist())};old_lookup={sid:i for i,sid in enumerate(sf['state_ids'].tolist())}
 low,high=action_bounds();neighbor_cache={};stability={}
 # K selection sees action covariance only, not any branch or future labels.
 for K in [10,20]:
  scores=[]
  for row in rows:
   i=actor_lookup[row['state_id']];poolids=np.flatnonzero((tasks==row['task_id'])&eligible)
   near=poolids[np.argsort(1-f['pool_features'][poolids]@f['state_features'][i])[:K]]
   assert len(near)==K
   x=f['pool_actions'][near,:5].reshape(K,35);traces=np.array([np.var(np.delete(x,j,axis=0),axis=0).sum() for j in range(K)])
   score=float(traces.std()/max(traces.mean(),1e-10));scores.append(score);neighbor_cache[(row['state_id'],K)]=near
  stability[str(K)]=dict(mean_relative_jackknife_trace_std=float(np.mean(scores)),median=float(np.median(scores)),states=len(rows))
 K=min([10,20],key=lambda k:(stability[str(k)]['mean_relative_jackknife_trace_std'],k))
 for name in ['states','episodes','invalid','candidates']: (D/name).mkdir(parents=True,exist_ok=True)
 selected=[];counts=collections.Counter();candidate_rows=[]
 for row in rows:
  sid=row['state_id'];i=actor_lookup[sid];v=old_lookup[sid];task=row['task_id'];th=thresholds[str(task)]
  with np.load(R/'data/action_sensitive_replay_v3/states'/f'{sid}.npz',allow_pickle=False) as s:
   base_raw=s['candidate_raw'][0].copy();base_norm=s['candidate_normalized'][0,:5].reshape(35).copy();assert s['valid_action_mask'].sum()==5
  near=neighbor_cache[(sid,K)];local=f['pool_actions'][near,:5].reshape(K,35).astype(np.float64)
  cov=np.cov(local,rowvar=False,ddof=0);std=local.std(0);eigen,vec=np.linalg.eigh(cov);direction=vec[:,-1]*np.sqrt(max(eigen[-1],0))
  # Sign is fixed independent of any outcome.
  if direction[np.argmax(abs(direction))]<0:direction=-direction
  oldids=np.flatnonzero(oldtask==task);oldnear=oldids[np.argsort(1-old['train_features'][oldids]@sf['features'][v])[:10]]
  supportpool=old['train_actions'][oldnear,:5].reshape(10,35)
  local_order=np.argsort(np.mean((local-base_norm)**2,axis=1))
  rng=np.random.default_rng(SEED+task*1000+row['initial_state_id']);noise=rng.normal(size=35)*std
  specifications=[]
  def add(name,norm,extra):
   raw=np.zeros((10,7),np.float32);raw[:5]=raw_action(norm.reshape(5,7),low,high)
   if name=='behavior':raw=base_raw.copy()
   raw[5:]=0;n=np.zeros((10,7),np.float32);n[:5]=normalized(raw[:5],low,high)
   bucket,distance=support_bucket(n[:5],supportpool,th)
   specifications.append(dict(name=name,raw=raw,norm=n,bucket=bucket,support_distance=distance,
    rms_from_behavior=float(np.sqrt(np.mean((n[:5].reshape(35)-base_norm)**2))),**extra))
  add('behavior',base_norm,dict(kind='behavior',scale=0.,direction=np.zeros(35).tolist()))
  # Choose a real nearest-neighbor endpoint and the smallest fixed lambda that
  # meets the unchanged support rule. Endpoint search is label-blind.
  used=set()
  for name,lambdas in [('interpolation_low',[.25,.50,.75,.875,.9375,1.]),('interpolation_high',[.50,.75,.875,.9375,1.])]:
   options=[]
   for order in local_order:
    if int(order) in used:continue
    endpoint=local[order]
    for lam in lambdas:
     norm=(1-lam)*base_norm+lam*endpoint
     actual=normalized(raw_action(norm.reshape(5,7),low,high),low,high).reshape(35)
     bucket,dist=support_bucket(actual,supportpool,th)
     options.append((bucket!='Near-ID',lam,float(np.sqrt(np.mean((actual-base_norm)**2))),int(order),norm,dist))
   chosen=min(options,key=lambda x:x[:4]);_,lam,_,order,norm,_=chosen;used.add(order)
   add(name,norm,dict(kind='interpolation',scale=lam,neighbor_index=int(near[order]),neighbor_episode_id=meta[int(near[order])]['episode_id'],direction=(local[order]-base_norm).tolist()))
  add('local_noise',base_norm+.10*noise,dict(kind='local_noise',scale=.10,direction=noise.tolist()))
  add('local_pca',base_norm+.20*direction,dict(kind='local_pca',scale=.20,direction=direction.tolist()))
  # Keep two mid-range directions. Fixed pre-outcome amplitude grid; retain
  # nearest available result when the Mid annulus cannot be reached.
  for name,sign in [('mid_pca_minus',-1),('mid_pca_plus',1)]:
   options=[]
   for size in [.30,.50,.75,1.,1.5,2.]:
    norm=base_norm+sign*size*direction;actual=normalized(raw_action(norm.reshape(5,7),low,high),low,high).reshape(35)
    bucket,dist=support_bucket(actual,supportpool,th)
    penalty=max(th['p50']-dist,0,dist-th['p90'])
    options.append((bucket!='Mid',penalty,size,norm))
   _,_,size,norm=min(options,key=lambda x:x[:3]);add(name,norm,dict(kind='mid_pca',scale=size,direction=(sign*direction).tolist()))
  assert len(specifications)==7
  raw=np.stack([s['raw'] for s in specifications]);norm=np.stack([s['norm'] for s in specifications]);padded=np.zeros((7,10,32),np.float32);padded[:,:,:7]=norm
  v4sid='v4_'+sid;path=D/'candidates'/f'{v4sid}.npz'
  np.savez_compressed(path,raw=raw,normalized=norm,padded32=padded,local_std=std.reshape(5,7),local_covariance=cov,
    neighbor_indices=near,neighbor_actions=local.reshape(K,5,7),support_neighbor_indices=oldnear,valid_mask=np.arange(10)<5)
  specs=[{k:v for k,v in s.items() if k not in ['raw','norm']} for s in specifications]
  for j,s in enumerate(specs): counts[s['bucket']]+=1;candidate_rows.append(dict(state_id=v4sid,task_id=task,candidate_index=j,**s))
  selected.append({**row,'state_id':v4sid,'v3_state_id':sid,'candidate_path':str(path),'candidate_sha256':sha(path),
    'candidate_specs':specs,'phase_selection_reused_without_outcome_selection':True,'branch_seed':SEED+1000000+task*1000+row['initial_state_id']})
 atomic(D/'state_selection_manifest.json',dict(records=selected,states=300,selection_before_v4_branches=True,no_v3_branch_labels_used=True))
 atomic(D/'local_manifold.json',dict(K_selected=K,stability=stability,selection_rule='Minimum mean relative jackknife covariance-trace STD, using real actions only',
  candidate_coverage_before_outcomes=dict(counts),thresholds=thresholds,records=candidate_rows))
 frozen=[R/'data/pilot_libero_v1/manifest.json',R/'data/action_sensitive_replay_v2/manifest.json',R/'data/action_sensitive_replay_v3/manifest.json',
  R/'data/action_sensitive_replay_v3/state_selection_manifest.json',R/'artifacts/counterfactual_final.parquet',R/'artifacts/counterfactual_results_v2.json',
  R/'artifacts/replay_action_audit_v3/ood_analysis.json',R/'artifacts/v3_positive_control.json',R/'configs/splits/libero_factorial_v1.json']
 frozen += list((R/'reports').glob('*v3*.md'))
 protocol=dict(dataset_version='action_sensitive_replay_v4',seed=SEED,target_states=300,actions_per_state=7,target_outcomes=2100,
  split_hash=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text())['sha256'],gamma=.9999,horizon=10,replan_steps=5,action_dim=7,padded_dim=32,restore_gate=1e-5,
  states='Exact v3 preselected train anchors, all retained; no v3 branch outcomes used; phases/fallbacks preserved',
  K=K,noise_scale=.10,near_pca_scale=.20,interpolation_lambda_grids=[[.25,.50,.75,.875,.9375,1.],[.50,.75,.875,.9375,1.]],
  mid_pca_scale_grid=[.30,.50,.75,1.,1.5,2.],geometry_adaptation='Fixed support-only selection BEFORE any v4 outcomes; same candidate count for all states',
  support_definition='UNCHANGED original200train public-SigLIP ten-state neighbor action RMS; original per-task p50/p90/p99 thresholds. New actor features only choose local generation neighbors.',
  candidate_selection_sha256=sha(D/'state_selection_manifest.json'),local_manifold_sha256=sha(D/'local_manifold.json'),
  positive_control=dict(folds=5,hidden=[128,64],pca=32,updates=1000,batch_states=64,lr=.001,rank_loss_weight=.1,rank_temperature=.1,
   tie_threshold=1e-6,near_effective_state_definition='At least one within-Near-ID action pair with |deltaG|>1e-6; all seven outcomes of each(task,initialID) stay together',
   primary_probe='B',probe_A='Actual frozen pi05 visual embedding + action',probe_B='Same embedding + explicit8Dproprio + action',
   probe_C='Explicit8Dproprio + action',probe_D='Current GT geometry/contact/grasp + proprio + action; diagnostics ONLY',
   gate='Primary B overall>=.60 CIlo>.50; Near-ID>=.65 with>=100informative held-out states; overall Spearman CIlo>0',
   representation_case_C='GT - primary B paired rank >=.10 and paired CIlo>0, GT overall>=.65; GT never authorizes formal training'),
  replay_mixes={'R1':{'deployment':.75,'branch':.25},'R2':{'deployment':.50,'branch':.50}},primary_mix='R1',formal_steps=2000,checkpoints=[250,500,1000,2000],
  actor_updates=False,beta_dynamic=False,GT_formal_input=False,frozen_input_hashes={str(p):sha(p) for p in frozen},
  code_hashes={p.name:sha(p) for p in (R/'scripts').glob('*v4*.py')})
 atomic(D/'protocol.json',protocol)
 report='# V4 candidate generation (frozen before outcomes)\n\n'+json.dumps(protocol,indent=2)+'\n\n## Offline K stability and geometric coverage\n\n'+json.dumps(dict(stability=stability,K=K,counts=dict(counts)),indent=2)
 report+='\n\nBehavior local-noise and local-PCA remain strictly centered on behavior; they are not projected or relabeled as Near-ID. Interpolation may need lambda above0.75 when behavior itself is outside old support; those endpoints and lambdas are recorded. Mid candidates use a fixed geometry-only grid. No source state is excluded for geometry or future labels.\n'
 (REP/'v4_candidate_generation.md').write_text(report)
 atomic(O/'offline_complete.json',dict(K=K,coverage=dict(counts),candidate_config_hash=sha(D/'protocol.json'),states=300,outcomes_planned=2100))
 print('V4_PROTOCOL_FROZEN',sha(D/'protocol.json'),json.dumps(dict(K=K,counts=dict(counts),stability=stability)),flush=True)

if __name__=='__main__':main()
