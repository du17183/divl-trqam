# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Preregistered train-only neighborhood audit, dev support audit, diagnostic CV."""
from pathlib import Path
import json,itertools,time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr,spearmanr
import torch
from torch import nn

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3';REP=R/'reports'
SEED=2026100717;TIE=1e-6

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [safe(v) for v in x]
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(x):return json.dumps(safe(x),indent=2,allow_nan=False)
def stats(x):
    x=np.asarray(x,dtype=float);return dict(mean=x.mean(),std=x.std(),p10=np.quantile(x,.1),p50=np.quantile(x,.5),p90=np.quantile(x,.9),min=x.min(),max=x.max()) if len(x) else None
def metrics(q,g):
    pairs=[(i,j) for i,j in itertools.combinations(range(len(g)),2) if abs(g[i]-g[j])>TIE]
    rank=np.mean([.5 if q[i]==q[j] else float((q[i]-q[j])*(g[i]-g[j])>0) for i,j in pairs]) if pairs else np.nan
    rho=spearmanr(q,g).statistic if len(g)>1 and np.ptp(g)>TIE and np.ptp(q)>1e-10 else np.nan
    return dict(rank=rank,spearman=rho,pairs=len(pairs),candidates=len(g))
def clustered(rows,key):
    usable=[v for v in rows if np.isfinite(v[key])]
    if not usable:return dict(states=0,mean=None,ci95=None)
    tasks=sorted({v['task_id'] for v in usable});rng=np.random.default_rng(SEED);boot=[]
    for _ in range(3000):
        chosen=[]
        for t in tasks:
            part=[v[key] for v in usable if v['task_id']==t]
            chosen.extend(rng.choice(part,size=len(part),replace=True))
        boot.append(np.mean(chosen))
    return dict(states=len(usable),mean=np.mean([v[key] for v in usable]),median=np.median([v[key] for v in usable]),ci95=np.quantile(boot,[.025,.975]),bootstrap='3000 whole-state resamples within task; conditional on informative subset')

def main():
    torch.set_num_threads(2)
    f=np.load(O/'public_features.npz');z=f['train_features'];dz=f['dev_features'];actions=f['train_actions'];masks=f['train_masks'];state=f['train_proprio']
    meta=json.loads(str(f['train_metadata']));dev=json.loads(str(f['dev_metadata']));task=np.array([v['task_id'] for v in meta]);ep=np.array([v['episode_id'] for v in meta]);g=np.array([v['future_return'] for v in meta]);success=np.array([v['success'] for v in meta]);remaining=np.array([v['remaining_steps'] for v in meta]);low=f['action_low'];high=f['action_high']
    futures=[];actrows=[];pairrows=[];cal=[];neighbor_map={}
    for t in range(10):
        ids=np.flatnonzero(task==t);dist=cdist(z[ids],z[ids],metric='cosine');np.fill_diagonal(dist,np.inf)
        for local,i in enumerate(ids):
            dist[local,ep[ids]==ep[i]]=np.inf
            order=ids[np.argsort(dist[local])]
            for k in [10,20]:
                n=order[:k];neighbor_map[(int(i),k)]=n
                aq=actions[n,:5];common=int(min(masks[i].sum(),np.min(masks[n].sum()),5));vec=aq[:,:common].reshape(k,-1)
                cov=np.cov(vec,rowvar=False,ddof=0)
                arow=dict(state_index=int(i),task_id=t,K=k,valid_steps=common,variance_mean=float(np.var(vec,axis=0).mean()),first_action_variance=float(np.var(aq[:,0],axis=0).mean()),
                    first5_variance=float(np.var(aq,axis=0).mean()),per_dimension_std=np.std(aq[:,:common],axis=(0,1)).tolist(),
                    covariance_per_dimension=np.cov(aq[:,:common].mean(1),rowvar=False,ddof=0).tolist(),covariance_trace=float(np.trace(cov)),arm_variance=float(np.var(aq[:,:common,:6],axis=0).mean()),gripper_variance=float(np.var(aq[:,:common,6],axis=0).mean()),
                    mean_neighbor_action_rms=float(np.mean([np.sqrt(np.mean((actions[i,:common]-actions[j,:common])**2)) for j in n])))
                actrows.append(arow)
                futures.append(dict(state_index=int(i),task_id=t,K=k,variance=float(np.var(g[n])),return_range=float(np.ptp(g[n])),all_success=bool(success[n].all()),all_failure=bool((~success[n]).all()),mixed=bool(success[n].any() and (~success[n]).any()),remaining_steps_mean=float(remaining[n].mean()),remaining_steps_std=float(remaining[n].std())))
                if k==10:
                    rms=[np.sqrt(np.mean((actions[i,:common]-actions[j,:common])**2)) for j in n]
                    cal.append(dict(task_id=t,state_index=int(i),local_knn_action_rms=float(min(rms))))
                    for j,d in zip(n,rms):pairrows.append(dict(task_id=t,left=int(i),right=int(j),visual_distance=float(1-z[i]@z[j]),action_rms=float(d),return_difference=float(abs(g[i]-g[j])),outcome_different=bool(success[i]!=success[j])))
        print('NEIGHBORS',t,flush=True)
    diversity={};future_summary={};pair_summary={}
    for k in [10,20]:
        for t in list(range(10))+['all']:
            aa=[x for x in actrows if x['K']==k and (t=='all' or x['task_id']==t)];ff=[x for x in futures if x['K']==k and (t=='all' or x['task_id']==t)]
            diversity[f'{t}_K{k}']={name:stats([x[name] for x in aa]) for name in ['variance_mean','first_action_variance','first5_variance','arm_variance','gripper_variance','mean_neighbor_action_rms']}
            future_summary[f'{t}_K{k}']=dict(states=len(ff),future_variance=stats([x['variance'] for x in ff]),return_range=stats([x['return_range'] for x in ff]),all_success_ratio=np.mean([x['all_success'] for x in ff]),all_failure_ratio=np.mean([x['all_failure'] for x in ff]),mixed_ratio=np.mean([x['mixed'] for x in ff]))
    examples=[]
    for t in range(10):
        p=[x for x in pairrows if x['task_id']==t];vd=np.array([x['visual_distance'] for x in p]);ad=np.array([x['action_rms'] for x in p]);gd=np.array([x['return_difference'] for x in p]);threshold=dict(visual_p10=float(np.quantile(vd,.1)),action_p50=float(np.quantile(ad,.5)),nonzero_future_p50=float(np.quantile(gd[gd>TIE],.5)) if (gd>TIE).any() else None)
        chosen=[x for x in p if x['visual_distance']<=threshold['visual_p10'] and x['action_rms']>=threshold['action_p50'] and threshold['nonzero_future_p50'] is not None and x['return_difference']>threshold['nonzero_future_p50']]
        pair_summary[t]=dict(pairs=len(p),thresholds=threshold,similar_state_different_action_future_pairs=len(chosen),fraction=len(chosen)/len(p),mixed_outcome_pair_ratio=np.mean([x['outcome_different'] for x in p]),pearson_action_distance_vs_future_difference=pearsonr(ad,gd).statistic if np.std(gd)>0 else None,spearman=spearmanr(ad,gd).statistic if np.std(gd)>0 else None)
        examples.extend(sorted(chosen,key=lambda x:x['visual_distance'])[:3])
    pd.DataFrame(pairrows).to_csv(O/'neighborhood_pairs.csv',index=False)
    for name,x in [('action_diversity',dict(summary=diversity,records=actrows)),('future_diversity',dict(summary=future_summary,records=futures,pair_summary=pair_summary,examples=examples)),('ood_calibration',cal)]: (O/f'{name}.json').write_text(dump(x))
    cf=pd.read_parquet(R/'artifacts/counterfactual_final.parquet');cf=cf[cf.kind=='candidate'].copy();oods=[];thresholds={}
    for t in range(10):
        c=[x['local_knn_action_rms'] for x in cal if x['task_id']==t];thresholds[t]=dict(zip(['p50','p90','p99'],np.quantile(c,[.5,.9,.99]).tolist()))
    for n,m in enumerate(dev):
        ids=np.flatnonzero(task==m['task_id']);dist=cdist(dz[n:n+1],z[ids],metric='cosine')[0];neighbors=ids[np.argsort(dist)[:10]];k=m['k'];pool=actions[neighbors,:k].reshape(10,-1);mu=pool.mean(0);global_std=actions[ids,:k].reshape(len(ids),-1).std(0);std=np.maximum(pool.std(0),.1*np.maximum(global_std,1e-3));cov=np.cov(pool,rowvar=False,ddof=0);cov+=np.diag(.1*np.maximum(global_std**2,1e-6));inv=np.linalg.pinv(cov);subset=cf[cf.state_id==m['state_id']].sort_values('candidate_index');native=np.stack([np.stack(x) for x in subset.action]);candidate=2*(native-low)/(high-low+1e-6)-1;candidate=candidate[:,:k].reshape(8,-1);behavior=candidate[0]
        for j,a in enumerate(candidate):
            local=float(np.sqrt(np.mean((pool-a)**2,axis=1)).min());same=float(np.sqrt(np.mean((actions[ids,:k].reshape(len(ids),-1)-a)**2,axis=1)).min());th=thresholds[m['task_id']];bucket='Near-ID' if local<=th['p50'] else 'Mid' if local<=th['p90'] else 'Far-OOD'
            row=subset.iloc[j];oods.append(dict(state_id=m['state_id'],task_id=m['task_id'],candidate_index=j,bucket=bucket,extreme_p99=bool(local>th['p99']),local_action_rms=local,same_task_nearest_action_rms=same,behavior_rms=float(np.sqrt(np.mean((a-behavior)**2))),mahalanobis_like=float(np.sqrt(max((a-mu)@inv@(a-mu),0)/len(a))),mean_abs_z=float(np.mean(abs((a-mu)/std))),max_abs_z=float(np.max(abs((a-mu)/std))),future_return=row.future_return,q_scalar=row.q_scalar,q_divl=row.q_divl,success=bool(row.success)))
    support=pd.DataFrame(oods);support.to_csv(O/'counterfactual_support.csv',index=False);ranking={}
    for b in ['all','Near-ID','Mid','Far-OOD']:
        part=support if b=='all' else support[support.bucket==b];ranking[b]={}
        for method in ['scalar','divl']:
            rows=[]
            for sid,p in part.groupby('state_id'):
                mm=metrics(p['q_'+method].values,p.future_return.values);mm.update(state_id=sid,task_id=int(p.task_id.iloc[0]));rows.append(mm)
            ranking[b][method]=dict(candidates=len(part),informative_pairs=sum(x['pairs'] for x in rows),rank=clustered(rows,'rank'),spearman=clustered(rows,'spearman'),states=rows)
    ood=dict(calibration=thresholds,counts=support.bucket.value_counts().to_dict(),extreme_p99=int(support.extreme_p99.sum()),ranking=ranking,limitation='All categories estimate within-category ranking. States with <2 distinct-return candidates give no ranking evidence; sparse categories are not called random. Coverage uses observational public-feature neighborhoods, not exact physical-state support.')
    (O/'ood_analysis.json').write_text(dump(ood));print('OOD',dump(dict(counts=ood['counts'],rank={b:{m:v['rank'] for m,v in x.items()} for b,x in ranking.items()})),flush=True)
    # Disposable diagnostic: all candidate actions of a state stay in the same fold.
    ordered=support.sort_values(['state_id','candidate_index']);state_ids=[m['state_id'] for m in dev];native=[];returns=[]
    for m in dev:
        p=cf[cf.state_id==m['state_id']].sort_values('candidate_index');native.append(np.stack([np.stack(a) for a in p.action])[:,:5]);returns.append(p.future_return.values)
    aa=np.stack(native);aa=2*(aa-low)/(high-low+1e-6)-1;aa=np.where(np.arange(5)[None,None,:,None]<np.array([m['k'] for m in dev])[:,None,None,None],aa,0).reshape(50,8,-1).astype(np.float32);gg=np.stack(returns).astype(np.float32)
    proprio=np.stack([np.load(R/'data/action_diagnostic_v1'/f'{m["state_id"]}_scalar_step2000.npz')['proprio'] for m in dev]);tasks=np.array([m['task_id'] for m in dev]);folds=np.zeros(50,int)
    for t in range(10):folds[np.flatnonzero(tasks==t)]=np.arange(5)
    predictions=np.zeros((50,8));records=[]
    for fold in range(5):
        torch.manual_seed(SEED+fold);tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
        center=dz[tr].mean(0);_,_,v=np.linalg.svd(dz[tr]-center,full_matrices=False);basis=v[:32].T
        zf=(dz[tr]-center)@basis;zt=(dz[te]-center)@basis;zstd=np.maximum(zf.std(0),1e-3);zf/=zstd;zt/=zstd
        sm=proprio[tr].mean(0);ss=np.maximum(proprio[tr].std(0),1e-3)
        def inp(ids,embedding):
            static=np.concatenate([embedding,(proprio[ids]-sm)/ss,np.eye(10)[tasks[ids]]],axis=1).astype(np.float32)
            return np.concatenate([np.repeat(static[:,None],8,axis=1),aa[ids]],axis=-1)
        xx=inp(tr,zf);xt=inp(te,zt);mean=xx.reshape(-1,xx.shape[-1]).mean(0);std=np.maximum(xx.reshape(-1,xx.shape[-1]).std(0),.05);x=torch.tensor((xx-mean)/std);y=torch.tensor(gg[tr]);test=torch.tensor((xt-mean)/std)
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1));opt=torch.optim.Adam(net.parameters(),lr=.001)
        ii,jj=np.triu_indices(8,1);ii=torch.tensor(ii);jj=torch.tensor(jj);difference=y[:,ii]-y[:,jj];valid=abs(difference)>TIE;sign=torch.sign(difference)
        for step in range(1000):
            q=net(x).squeeze(-1);reg=((q-y)**2).mean();rank=nn.functional.softplus(-sign*(q[:,ii]-q[:,jj])/.1)[valid].mean();loss=reg+.1*rank
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():predictions[te]=net(test).squeeze(-1).numpy();qtrain=net(x).squeeze(-1).numpy()
        trainrank=np.nanmean([metrics(qtrain[i],gg[tr[i]])['rank'] for i in range(len(tr))]);print('CV_FOLD',fold,'train_rank',trainrank,'heldout_rank',np.nanmean([metrics(predictions[i],gg[i])['rank'] for i in te]),flush=True)
        records.append(dict(fold=fold,train_state_ids=[state_ids[i] for i in tr],heldout_state_ids=[state_ids[i] for i in te],train_rank=trainrank,final_loss=float(loss),PCA_components=32,steps=1000))
    rows=[dict(state_id=state_ids[i],task_id=int(tasks[i]),**metrics(predictions[i],gg[i])) for i in range(50)]
    control=dict(protocol='5-fold whole-state CV, one held-out state/task/fold; all 8 actions travel together; train-only PCA32 and scaler; small128/64 MLP; 1000steps Adam .001 MSE+.1rank softplus/.1; no model/step selection on heldout',rank=clustered(rows,'rank'),spearman=clustered(rows,'spearman'),folds=records,states=rows,formal_training_use=False)
    (O/'positive_control.json').write_text(dump(control));np.savez_compressed(O/'positive_control_predictions.npz',prediction=predictions,returns=gg,folds=folds,state_ids=np.asarray(state_ids))
    print('POSITIVE_CONTROL',dump({k:v for k,v in control.items() if k in ['rank','spearman']}),flush=True)
    def write(name,title,body): (REP/name).write_text('# '+title+'\n\n'+body)
    write('current_replay_action_diversity.md','Current replay action diversity','All200 train episodes represented by five fixed temporal samples each (1000 states,100/task). Frozen public SigLIP; no learned critic embedding. Same-task/different-episode neighbors. Prefix is at most5 executed actions; nominalH10 masked remainder is not a real extra prefix.\n\n'+dump(diversity)+'\n\nFull per-state covariance/std in artifacts/replay_action_audit_v3/action_diversity.json. Neighborhoods are observational and may mix distinct physical states.\n')
    write('current_replay_future_diversity.md','Current replay future diversity',dump(future_summary)+'\n\n## Action/future relationship and empirical case thresholds\n\n'+dump(pair_summary)+'\n\n## Actual examples (indices map to public_features metadata)\n\n'+dump(examples)+'\n\nZero/nonzero outcome comparisons reported separately from timing differences. Observational correlation does not identify causal action effects.\n')
    write('counterfactual_ood_analysis.md','Frozen counterfactual support analysis',dump({k:v for k,v in ood.items() if k!='ranking'})+'\n\n## Ranking by support\n\n'+dump({b:{m:{k:v for k,v in val.items() if k!='states'} for m,val in x.items()} for b,x in ranking.items()})+'\n\nNo new rollout. Training leave-one-episode-out p50/p90/p99 thresholds preregistered before labels inspected; no candidate removal from original primary metrics.\n')
    write('action_learnability_positive_control.md','Disposable action learnability positive control',dump(control)+'\n\nDiagnostic dev-state CV only; none of these50 states/outcomes may enter formal v2 training. No trained critic encoder or candidate-index labels. PCA32 overrides the protocol initial64 because40 train states/fold imply maximum39 nontrivial components;32 chosen before fitting any fold. A failure here cannot establish theoretical unlearnability: only50 states/8 mixed-outcome states, tiny latency differences, heterogeneous10tasks.\n')
    (O/'cpu_complete.json').write_text(dump(dict(status='complete',action_summary=diversity['all_K10'],future_summary=future_summary['all_K10'],ood_counts=ood['counts'],positive_control_rank=control['rank'],positive_control_spearman=control['spearman'])))

if __name__=='__main__':main()
