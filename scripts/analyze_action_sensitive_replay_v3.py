"""Complete retained-outcome audit plus whole-state CV, never train dev50."""
import itertools,json,numpy as np,pandas as pd,torch
from torch import nn
from action_sensitive_v3_common import R,D,O,SEED,sha,atomic
from analyze_replay_action_v3 import metrics,clustered,safe,stats
from evaluate_action_sensitive_replay_v2 import first_rows


def main():
    torch.set_num_threads(2);manifest=json.loads((D/'manifest.json').read_text());protocol=json.loads((D/'protocol.json').read_text())
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text());assert manifest['split_hash']==split['sha256']
    f=np.load(O/'v3_frozen_features.npz');ids=f['state_ids'].tolist();task=f['task'];initial=f['initial'];z=f['features'];p=f['proprio'];a=f['actions'];g=f['returns'];support=f['support']
    groups={};lookup={r['state_id']:{ } for r in manifest['records']}
    for r in manifest['records']:groups.setdefault(r['state_id'],[]).append(r)
    state_metrics=[];branch_metrics=[];baseline_max=0.;prefix_truncations=0
    for i,sid in enumerate(ids):
        rows=sorted(groups[sid],key=lambda x:x['candidate_index']);assert len(rows)==5
        with np.load(D/'states'/f'{sid}.npz',allow_pickle=False) as state:
            raw=state['candidate_raw'];norm=state['candidate_normalized'];mask=state['valid_action_mask']
            base,wrist,proprio=state['base'],state['wrist'],state['proprio']
            assert np.count_nonzero(state['candidate_padded32'][:,:,7:])==0
        for j,row in enumerate(rows):
            assert row['initial_state_id'] in split['tasks'][row['task_name']]['train']
            assert sha(row['path'])==row['sha256']
            first=first_rows(row['path'],['base','wrist','state','actions','action_mask'])
            assert np.array_equal(first['base'],base) and np.array_equal(first['wrist'],wrist) and np.array_equal(first['state'],proprio)
            actual=first['action_mask'];assert np.array_equal(first['actions'][actual],raw[j][actual])
            prefix_truncations+=int(actual.sum()<mask.sum())
            with np.load(row['path'],allow_pickle=False) as episode:
                rewards=episode['env_step_rewards'];computed=sum(.9999**k*float(r) for k,r in enumerate(rewards))
                assert abs(computed-row['future_return'])<1e-6
            if j==0:baseline_max=max(baseline_max,row['baseline_reproduction_error'])
            branch_metrics.append({**row,'support':str(support[i,j]),'delta_from_behavior':float(g[i,j]-g[i,0]),
                'absolute_delta_from_behavior':float(abs(g[i,j]-g[i,0])),
                'immediate_terminal_failure':not row['success'] and row['chunks']==1})
        pair_counts={str(t):sum(abs(float(g[i,x])-float(g[i,y]))>t for x,y in itertools.combinations(range(5),2)) for t in protocol['informative_thresholds']}
        successes=[r['success'] for r in rows]
        state_metrics.append(dict(state_id=sid,task_id=int(task[i]),phase=rows[0]['phase'],phase_fallback=rows[0]['phase_fallback'],
            initial_state_id=int(initial[i]),return_range=float(np.ptp(g[i])),return_variance=float(np.var(g[i])),
            effective_pairs=pair_counts,mixed_outcome=any(successes) and not all(successes)))
    pd.DataFrame(branch_metrics).to_parquet(R/'artifacts/v3_branch_metrics.parquet',index=False)
    summary=dict(states=len(ids),attempted_states=manifest['attempted_states'],invalid_states=manifest['invalid_states'],outcomes=len(branch_metrics),
        task_counts={str(t):int((task==t).sum()) for t in range(10)},
        candidate_counts={name:sum(r['candidate_name']==name for r in branch_metrics) for name in protocol['candidates']},
        support_counts={b:int((support==b).sum()) for b in ['Near-ID','Mid','Far-OOD']},
        informative={str(t):dict(states=sum(v['return_range']>t for v in state_metrics),
            proportion=float(np.mean([v['return_range']>t for v in state_metrics])),
            average_effective_pairs=float(np.mean([v['effective_pairs'][str(t)] for v in state_metrics]))) for t in protocol['informative_thresholds']},
        zero_return_difference_ratio=float(np.mean([v['return_range']==0 for v in state_metrics])),
        mixed_success_failure_states=sum(v['mixed_outcome'] for v in state_metrics),
        return_range=stats([v['return_range'] for v in state_metrics]),baseline_gate_max_error=baseline_max,
        actor_visible_inputs_bit_identical=True,preexecution_plan_features_only=True,postoutcome_prefix_truncations=prefix_truncations,
        all_valid_state_outcomes_retained=True,dev_states_used=False)
    perturbations={}
    for name in ['small','medium','large']:
        rows=[r for r in branch_metrics if r['candidate_name'].startswith(name)]
        perturbations[name]=dict(outcomes=len(rows),action_distance=stats([r['behavior_rms'] for r in rows]),
            absolute_delta_G=stats([r['absolute_delta_from_behavior'] for r in rows]),
            different_future_ratios={str(t):float(np.mean([r['absolute_delta_from_behavior']>t for r in rows])) for t in protocol['informative_thresholds']},
            success_ratio=float(np.mean([r['success'] for r in rows])),failure_ratio=float(np.mean([not r['success'] for r in rows])),
            immediate_failure_ratio=float(np.mean([r['immediate_terminal_failure'] for r in rows])),
            support_counts={b:sum(r['support']==b for r in rows) for b in ['Near-ID','Mid','Far-OOD']})
    atomic(O/'v3_action_supervision.json',safe(dict(summary=summary,perturbations=perturbations,states=state_metrics)))
    (R/'reports/v3_action_supervision_report.md').write_text('# V3 retained-outcome action supervision\n\n'+json.dumps(safe(dict(summary=summary,perturbations=perturbations)),indent=2)+'\n\nPhase predicates are generic physical-contact proxies. Selection was frozen before branching; no state removed for flatG or unsuccessful outcomes. Timing-only differences are separate from binary future mixtures. One continuation noise perstate/action does not identify a full conditional return law.\n')
    (R/'reports/v3_branch_collection_report.md').write_text('# V3 branch collection\n\n'+json.dumps(safe(summary),indent=2)+'\n\nComplete source and branch trajectories retain native env-step rewards, originalchunk rewards/discounts/terminal, preexecution planned7Dand padded32D actions, snapshots/controller/RNG state, RGB/state/instruction. Invalid restoration attempts are retained in data/action_sensitive_replay_v3/invalid.\n')
    predictions=np.zeros_like(g);folds=np.zeros(len(ids),int)
    for t in range(10):
        unique=sorted(set(initial[task==t]));assert len(unique)>=5
        for rank,init in enumerate(unique):folds[(task==t)&(initial==init)]=rank%5
    fold_records=[]
    for fold in range(5):
        torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold);tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
        center=z[tr].mean(0);_,_,v=np.linalg.svd(z[tr]-center,full_matrices=False);basis=v[:32].T
        ztr=(z[tr]-center)@basis;zte=(z[te]-center)@basis;zstd=np.maximum(ztr.std(0),1e-3);ztr/=zstd;zte/=zstd
        pm=p[tr].mean(0);ps=np.maximum(p[tr].std(0),1e-3)
        def inputs(indices,embedding):
            static=np.concatenate([embedding,(p[indices]-pm)/ps,np.eye(10)[task[indices]]],1).astype(np.float32)
            return np.concatenate([np.repeat(static[:,None],5,axis=1),a[indices]],-1)
        x=inputs(tr,ztr);xt=inputs(te,zte);mean=x.reshape(-1,x.shape[-1]).mean(0);std=np.maximum(x.reshape(-1,x.shape[-1]).std(0),.05)
        x=torch.tensor((x-mean)/std);xt=torch.tensor((xt-mean)/std);y=torch.tensor(g[tr])
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        opt=torch.optim.Adam(net.parameters(),lr=.001);ii,jj=np.triu_indices(5,1);ii=torch.tensor(ii);jj=torch.tensor(jj)
        for step in range(1000):
            sampled=torch.tensor(rng.integers(len(tr),size=64));q=net(x[sampled]).squeeze(-1);truth=y[sampled];delta=truth[:,ii]-truth[:,jj];valid=abs(delta)>1e-6
            rank=nn.functional.softplus(-torch.sign(delta)*(q[:,ii]-q[:,jj])/.1)[valid].mean() if valid.any() else q.sum()*0
            loss=((q-truth)**2).mean()+.1*rank;opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():predictions[te]=net(xt).squeeze(-1).numpy()
        rows=[metrics(predictions[i],g[i]) for i in te]
        fold_records.append(dict(fold=fold,train_states=len(tr),heldout_states=len(te),
            rank=float(np.nanmean([x['rank'] for x in rows])),spearman=float(np.nanmean([x['spearman'] for x in rows]))))
        print('V3_CONTROL_FOLD',json.dumps(safe(fold_records[-1])),flush=True)
    aggregates={}
    for bucket in ['all','Near-ID','Mid','Far-OOD']:
        records=[]
        for i,sid in enumerate(ids):
            indices=np.arange(5) if bucket=='all' else np.flatnonzero(support[i]==bucket)
            result=metrics(predictions[i,indices],g[i,indices]);result.update(state_id=sid,task_id=int(task[i]));records.append(result)
        aggregates[bucket]=dict(rank=clustered(records,'rank'),spearman=clustered(records,'spearman'),effective_pairs=sum(r['pairs'] for r in records),states=records)
    rank=aggregates['all']['rank'];passed=bool(rank['mean'] is not None and rank['mean']>=.60 and rank['ci95'][0]>.50)
    result=dict(passed=passed,engineering_gate=protocol['positive_control']['engineering_gate'],
        protocol=protocol['positive_control'],metrics=aggregates,folds=fold_records,
        state_split=True,initial_id_split=True,oracle_features=False,actor_updated=False,formal_model=False)
    atomic(R/'artifacts/v3_positive_control.json',safe(result));atomic(O/'positive_control_gate.json',dict(passed=passed,engineering_gate=result['engineering_gate']))
    np.savez_compressed(O/'v3_positive_predictions.npz',predictions=predictions,returns=g,folds=folds,state_ids=np.asarray(ids))
    (R/'reports/v3_positive_control_report.md').write_text('# V3 positive control\n\n'+json.dumps(safe(result),indent=2)+'\n\nNo candidate IDs, oracle physical state, success flags, remaining horizon, G or post-outcome mask are modelinputs. Both cameras are public frozen vision features. PCA/scalers are fit within each training fold. No checkpoint/model selection. Engineering gate is not a paper significance criterion.\n')
    print('V3_POSITIVE_CONTROL_COMPLETE',passed,json.dumps(safe({b:{k:v for k,v in x.items() if k!='states'} for b,x in aggregates.items()})),flush=True)


if __name__=='__main__':main()
