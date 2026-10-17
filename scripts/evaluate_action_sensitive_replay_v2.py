# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Describe all train interventions and test out-of-initial-state learnability."""
from pathlib import Path
import json,itertools,numpy as np
import zipfile
import torch
from torch import nn
from analyze_replay_action_v3 import metrics,safe,dump,stats,SEED
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3';D=R/'data/action_sensitive_replay_v2';P=R/'reports'

def first_rows(path,names):
    """Read only row0 of contiguousNPY entries; do not inflate fullRGB episodes."""
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in names:
            with archive.open(name+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    with np.load(path,allow_pickle=False) as f:result[name]=f[name][0].copy()
                    continue
                shape,fortran,dtype=reader(stream)
                if fortran or dtype.hasobject:
                    with np.load(path,allow_pickle=False) as f:result[name]=f[name][0].copy()
                    continue
                assert shape and shape[0]>0
                count=int(np.prod(shape[1:]));raw=stream.read(count*dtype.itemsize);assert len(raw)==count*dtype.itemsize
                result[name]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).copy()
    return result

def main():
    torch.set_num_threads(2);manifest=json.loads((D/'manifest.json').read_text());assert manifest['states']==1000 and manifest['outcomes']==5000
    feat=np.load(O/'public_features.npz');meta=json.loads(str(feat['train_metadata']));keys={(v['episode_id'],v['index']):i for i,v in enumerate(meta)};low=feat['action_low'];high=feat['action_high']
    groups={};source_actions={};early_prefix_terminations=0
    for row in manifest['records']:groups.setdefault(row['state_id'],[]).append(row)
    zz=[];pp=[];aa=[];gg=[];task=[];init=[];ids=[];groupstats=[];support=[]
    thresholds=json.loads((O/'ood_analysis.json').read_text())['calibration']
    for sid,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['candidate_index']);assert [v['candidate_index'] for v in rows]==list(range(5));source=rows[0]
        n=keys[(source['source_episode_id'],source['source_index'])];zz.append(feat['train_features'][n]);pp.append(feat['train_proprio'][n]);g=np.array([v['future_return'] for v in rows]);gg.append(g);task.append(source['task_id']);init.append(source['initial_state_id']);ids.append(sid)
        acts=[];first_obs=None
        for row in rows:
            if row['source_path'] not in source_actions:
                with np.load(row['source_path'],allow_pickle=False) as original:source_actions[row['source_path']]=original['actions']
            planned=source_actions[row['source_path']][row['source_index']].copy();k=row['k']
            if row['candidate_index']:
                size,sign={1:(.25,-1),2:(.25,1),3:(.5,-1),4:(.5,1)}[row['candidate_index']]
                direction=np.random.default_rng(row['branch_seed']).normal(size=(1,7)).astype(np.float32);scale=np.asarray(row['perturb_scale'],np.float32)
                bn=2*(planned-low)/(high-low+1e-6)-1
                planned=np.clip((bn+size*sign*direction*scale+1)/2*(high-low+1e-6)+low,-1,1).astype(np.float32);planned[k:]=0
            e=first_rows(row['path'],['actions','action_mask','base','wrist','state'])
            actual_k=int(e['action_mask'].sum());assert actual_k<=k
            assert np.array_equal(e['actions'][:actual_k],planned[:actual_k]),'Reconstructed planned prefix differs from actual controller actions'
            early_prefix_terminations+=int(actual_k<k)
            # Post-outcome truncation is not a valid feature for predicting
            # that outcome. Use the known pre-execution candidate plan and
            # common source mask, exactly as in the frozen dev400 probe.
            a=2*(planned-low)/(high-low+1e-6)-1;m=np.arange(len(planned))<k;acts.append(np.where(m[:,None],a,0)[:5].reshape(-1))
            signature=(e['base'].tobytes(),e['wrist'].tobytes(),e['state'].tobytes())
            if first_obs is None:first_obs=signature
            else:assert signature==first_obs,'Branches must have identical actor-visible input'
            assert row['initial_state_error']==0
            th=thresholds[str(row['task_id'])];v=row['local_action_rms'];support.append('Near-ID' if v<=th['p50'] else 'Mid' if v<=th['p90'] else 'Far-OOD')
        aa.append(np.stack(acts));outcomes=[v['success'] for v in rows]
        groupstats.append(dict(state_id=sid,task_id=source['task_id'],initial_state_id=source['initial_state_id'],future_variance=float(g.var()),return_range=float(np.ptp(g)),action_variance=float(np.var(acts,axis=0).mean()),different_future=bool(np.ptp(g)>1e-6),mixed_outcome=bool(any(outcomes) and not all(outcomes)),all_success=all(outcomes),all_failure=not any(outcomes)))
    zz=np.stack(zz);pp=np.stack(pp);aa=np.stack(aa).astype(np.float32);gg=np.stack(gg).astype(np.float32);task=np.array(task);init=np.array(init)
    summary=dict(states=1000,outcomes=5000,successes=sum(v['success'] for v in manifest['records']),different_future_states=sum(v['different_future'] for v in groupstats),mixed_success_failure_states=sum(v['mixed_outcome'] for v in groupstats),
        future_variance=stats([v['future_variance'] for v in groupstats]),action_variance=stats([v['action_variance'] for v in groupstats]),return_range=stats([v['return_range'] for v in groupstats]),support_counts={b:support.count(b) for b in ['Near-ID','Mid','Far-OOD']},
        baseline_gate_max_error=max(v['baseline_reproduction_error'] or 0 for v in manifest['records']),same_state_visible_input_exact=True,all_outcomes_retained=True,
        positive_control_actions='pre-execution planned candidates reconstructed from fixed original native action/registered seed and sigma; verified bit-equal on every actually executed prefix',early_prefix_terminations=early_prefix_terminations,post_outcome_mask_used_as_control_feature=False)
    data_gate=dict(passed=summary['different_future_states']>=250 and summary['same_state_visible_input_exact'] and summary['baseline_gate_max_error']<1e-5,
        conditions='all1000 baseline restoration gates; all5000 preregistered outcomes; >25%states have real non-tied future differences; actor-visible observations identical within each branch group; no dev states',summary=summary)
    (O/'replay_v2_data_gate.json').write_text(dump(data_gate));(O/'replay_v2_statistics.json').write_text(dump(dict(summary=summary,states=groupstats)))
    print('REPLAY_V2_DATA_GATE',dump(data_gate),flush=True)
    folds=np.zeros(1000,int)
    for t in range(10):
        unique=sorted(set(init[task==t]));assert len(unique)==10
        for rank,i in enumerate(unique):folds[(task==t)&(init==i)]=rank%5
    predictions=np.zeros_like(gg);foldrecords=[]
    for fold in range(5):
        torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold);tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
        center=zz[tr].mean(0);_,_,v=np.linalg.svd(zz[tr]-center,full_matrices=False);basis=v[:32].T;zf=(zz[tr]-center)@basis;zt=(zz[te]-center)@basis;zs=np.maximum(zf.std(0),1e-3);zf/=zs;zt/=zs
        pm=pp[tr].mean(0);ps=np.maximum(pp[tr].std(0),1e-3)
        def inp(ii,z):
            static=np.concatenate([z,(pp[ii]-pm)/ps,np.eye(10)[task[ii]]],axis=1).astype(np.float32)
            return np.concatenate([np.repeat(static[:,None],5,axis=1),aa[ii]],axis=-1)
        x=inp(tr,zf);test=inp(te,zt);mean=x.reshape(-1,x.shape[-1]).mean(0);std=np.maximum(x.reshape(-1,x.shape[-1]).std(0),.05);x=torch.tensor((x-mean)/std);test=torch.tensor((test-mean)/std);y=torch.tensor(gg[tr])
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1));opt=torch.optim.Adam(net.parameters(),lr=.001);ii,jj=np.triu_indices(5,1);ii=torch.tensor(ii);jj=torch.tensor(jj)
        for step in range(1000):
            sampled=torch.tensor(rng.integers(len(tr),size=64));q=net(x[sampled]).squeeze(-1);truth=y[sampled];diff=truth[:,ii]-truth[:,jj];valid=abs(diff)>1e-6
            rank=nn.functional.softplus(-torch.sign(diff)*(q[:,ii]-q[:,jj])/.1)[valid].mean() if valid.any() else q.sum()*0
            loss=((q-truth)**2).mean()+.1*rank;opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():predictions[te]=net(test).squeeze(-1).numpy()
        rows=[metrics(predictions[i],gg[i]) for i in te];foldrecords.append(dict(fold=fold,train_states=len(tr),heldout_states=len(te),ranking=float(np.nanmean([v['rank'] for v in rows])),spearman=float(np.nanmean([v['spearman'] for v in rows]))));print('V2_POSITIVE_FOLD',dump(foldrecords[-1]),flush=True)
    records=[dict(state_id=ids[i],task_id=int(task[i]),initial_state_id=int(init[i]),**metrics(predictions[i],gg[i])) for i in range(1000)]
    rng=np.random.default_rng(SEED);aggregate={}
    for metric in ['rank','spearman']:
        values={}
        for row in records:
            if np.isfinite(row[metric]):values.setdefault((row['task_id'],row['initial_state_id']),[]).append(row[metric])
        boots=[]
        for _ in range(3000):
            sample=[]
            for t in range(10):
                keys=[key for key in values if key[0]==t]
                if not keys:continue
                chosen=rng.integers(len(keys),size=len(keys));sample.extend([v for j in chosen for v in values[keys[j]]])
            boots.append(np.mean(sample))
        aggregate[metric]=dict(mean=np.nanmean([r[metric] for r in records]),ci95=np.quantile(boots,[.025,.975]),initial_state_clusters=len(values),informative_states=sum(np.isfinite(r[metric]) for r in records))
    result=dict(protocol='5 folds by whole train initial ID within task; both policy repetitions and all temporal anchors/candidates stay together. Train-only PCA32/scalers. Fixed128/64 MLP,1000 Adam .001 updates B64states, MSE+.1ranking. Pre-execution action plan, common source mask; no post-outcome truncation feature, candidate IDs or outcome features. Disposable diagnostic only.',metrics=aggregate,folds=foldrecords,formal_model=False)
    (O/'replay_v2_positive_control.json').write_text(dump(result));np.savez_compressed(O/'replay_v2_positive_predictions.npz',predictions=predictions,returns=gg,folds=folds,state_ids=np.asarray(ids))
    (P/'action_sensitive_replay_v2_report.md').write_text('# Action-sensitive replay v2\n\n'+dump(summary)+'\n\n## Data gate\n\n'+dump(data_gate)+'\n\n## Initial-state grouped learnability control\n\n'+dump(result)+'\n\nAll outcomes retained. No actor/QAM/DIVL update occurred during collection or this diagnostic. Native binary rewards unchanged. Timing variation and mixed success/failure variation are reported separately.\n')
    print('REPLAY_V2_ANALYSIS_COMPLETE',dump(result),flush=True)

if __name__=='__main__':main()
