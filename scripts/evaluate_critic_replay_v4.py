# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Reuse the exact frozen50/400 true candidates; new checkpoints, new directions."""
from pathlib import Path
import json,itertools,os
import numpy as np
import flax.nnx as nnx
import jax,jax.numpy as jnp
import orbax.checkpoint as ocp
import pandas as pd
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config
from analyze_replay_action_v3 import metrics,clustered,dump
from critic_v2_io import restore_critic
from critic_eval_pairing import paired_state_differences
R=Path('/opt/divl-trqam');O=R/'artifacts/action_sensitive_replay_v4'

def main():
    mix=os.environ.get('V4_EVAL_MIX','R1');assert mix in ['R1','R2']
    truth=pd.read_parquet(R/'artifacts/counterfactual_final.parquet');truth=truth[truth.kind=='candidate'];support=pd.read_csv(R/'artifacts/replay_action_audit_v3/counterfactual_support.csv')
    dev=sorted([v['metadata'] for v in json.loads((R/'artifacts/counterfactual_results_v2.json').read_text())],key=lambda m:m['state_id'])
    norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero');low=np.asarray(norm['actions'].q01)[:7];high=np.asarray(norm['actions'].q99)[:7];cfg=encoder_config(R);tokenizer=DIVLTokenizer(cfg)
    field=['base','wrist','state','actions','action_mask','rewards','discounts','prompt'];sample=[]
    for m in dev:
        with np.load(m['episode_path'],allow_pickle=False) as e:sample.append(({k:e[k] for k in field},m['index']))
    @nnx.jit
    def encode(model,obs):return model.encoder(**obs,train=False)
    @nnx.jit
    def predict(head,z,a,mask):return jnp.minimum(*head(z,a,mask))
    @nnx.jit
    def gradient(head,z,a,mask):return jax.grad(lambda x:jnp.minimum(*head(z,x,mask)).sum())(a)
    allmetrics={};staticdir=R/'data/critic_replay_v4_diagnostic'/mix;staticdir.mkdir(parents=True,exist_ok=True)
    selected=os.environ.get('V4_EVAL_METHOD');selected_step=os.environ.get('V4_EVAL_STEP');aggregate_only=os.environ.get('V4_EVAL_AGGREGATE')=='1'
    if selected:assert selected in ['scalar','divl'] and int(selected_step) in [250,500,1000,2000]
    if aggregate_only:
        allmetrics={m:{step:json.loads((O/f'v4_eval_{mix}_{m}_step{step:04d}.json').read_text()) for step in [250,500,1000,2000]} for m in ['scalar','divl']}
    methods=[] if aggregate_only else [selected] if selected else ['scalar','divl']
    for method in methods:
        allmetrics[method]={}
        for step in [int(selected_step)] if selected else [250,500,1000,2000]:
            critic=build_critic(LiberoTrainingConfig(value_type=method),cfg)
            critic=restore_critic(critic,R/'checkpoints/critic_replay_v4'/mix/method/f'step{step:04d}')
            results={b:[] for b in ['all','Near-ID','Mid','Far-OOD']}
            for n,m in enumerate(dev):
                batch=make_batch([sample[n]],tokenizer,norm,td_steps=1);z=encode(critic,batch['obs'])
                old=np.load(R/'data/action_diagnostic_v1'/f'{m["state_id"]}_scalar_step2000.npz');native=old['candidate_actions'];mask=old['action_mask'];a=np.where(mask[None,:,None],2*(native-low)/(high-low+1e-6)-1,0).astype(np.float32)
                q=np.asarray(predict(critic.q_head,jnp.repeat(z,8,axis=0),jnp.asarray(a),jnp.asarray(np.repeat(mask[None],8,axis=0))))
                part=truth[truth.state_id==m['state_id']].sort_values('candidate_index');g=part.future_return.values
                buckets=support[support.state_id==m['state_id']].sort_values('candidate_index').bucket.values
                for b in results:
                    ids=np.arange(8) if b=='all' else np.flatnonzero(buckets==b);result=metrics(q[ids],g[ids]);result.update(state_id=m['state_id'],task_id=m['task_id']);results[b].append(result)
                if step==2000:
                    grad=np.asarray(gradient(critic.q_head,z,jnp.asarray(a[0:1]),jnp.asarray(mask[None])))[0];unit=grad/max(float(np.linalg.norm(grad)),1e-10);out=dict(metadata=np.asarray(json.dumps(m)),candidate_actions=native,candidate_q=q,action_mask=mask,gradient=grad,proprio=old['proprio'])
                    for eps in [.01,.03,.05]:
                        for sign in [-1,1]:
                            na=a[0]+sign*eps*unit;raw=np.clip((na+1)/2*(high-low+1e-6)+low,-1,1).astype(np.float32);raw[~mask]=0;out[f'gradient_{eps}_{sign}']=raw
                    np.savez_compressed(staticdir/f'{m["state_id"]}_{method}.npz',**out)
            aggregate={b:dict(rank=clustered(v,'rank'),spearman=clustered(v,'spearman'),effective_pairs=sum(x['pairs'] for x in v),states=v) for b,v in results.items()};allmetrics[method][step]=aggregate
            (O/f'v4_eval_{mix}_{method}_step{step:04d}.json').write_text(dump(aggregate))
            print('V4_FIXED50',method,step,dump({b:{k:v for k,v in x.items() if k!='states'} for b,x in aggregate.items()}),flush=True)
    if selected:return
    paired={}
    for step in [250,500,1000,2000]:
        paired[step]={}
        for bucket in ['all','Near-ID','Mid','Far-OOD']:
            s=allmetrics['scalar'][step][bucket]['states'];d=allmetrics['divl'][step][bucket]['states']
            diff=paired_state_differences(s,d)
            paired[step][bucket]=dict(rank=clustered(diff,'rank'),spearman=clustered(diff,'spearman'),total_states=len(diff),states=diff,missing_estimate_policy='Preserve all states; bootstrap each metric only where both methods have a finite estimate, unchanged informative-state convention')
    prereq={}
    for m in ['scalar','divl']:
        v=allmetrics[m][2000];rank=v['all']['rank'];rho=v['all']['spearman'];near=v['Near-ID']['rank']
        conditions=dict(ranking=rank['mean'] is not None and rank['mean']>=.60 and rank['ci95'][0]>.5,spearman=rho['mean'] is not None and rho['ci95'][0]>0,near_id=near['mean'] is not None and near['mean']>=.60 and near['ci95'][0]>.5)
        prereq[m]=dict(conditions=conditions,ready_for_new_gradient=all(conditions.values()),actor_qam_allowed=False,reason='New critic gradients require real branches; old600 outcomes apply only to old directions')
    (O/f'v4_counterfactual_{mix}.json').write_text(dump(dict(primary_budget=2000,all_budgets=allmetrics,paired_divl_minus_scalar=paired,prerequisites=prereq,candidates_and_truth_unchanged=True,new_candidate_rollouts=0)))
    text='# V3 frozen counterfactual evaluation\n\nExact original50states/400candidate native actions/true returns. Original support bins retained. No new candidate rollout or checkpoint selection; primary2000 fixed in advance. Old600 gradient directions are not evidence for new critics.\n\n'+dump(prereq)+'\n\n'
    for m in ['scalar','divl']:
        report='# '+m+' v3 critic\n\n'
        for step,a in allmetrics[m].items():report+=f'## step{step}\n\n'+dump({b:{k:v for k,v in x.items() if k!='states'} for b,x in a.items()})+'\n\n'
        (R/'reports'/f'{m}_v4_{mix}_critic_report.md').write_text(report);text+=report+'\n'
    text+='\n## Paired DIVL-minus-Scalar\n\nWhole-state, task-stratified bootstrap of the paired state differences; no action-pair pseudo-replication.\n\n'+dump(paired)+'\n'
    (R/'reports'/f'v4_heldout_{mix}.md').write_text(text)
    print('V4_EVALUATION_COMPLETE',dump(prereq),flush=True)

if __name__=='__main__':main()
