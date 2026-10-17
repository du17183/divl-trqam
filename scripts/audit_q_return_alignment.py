# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Read-only current/next chunk alignment and actual retained critic predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from scipy.stats import pearsonr, spearmanr
from openpi import transforms
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.models.divl_losses import categorical_quantile, adaptive_quantile_level
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig, build_critic, encoder_config

ROOT=Path('/opt/divl-trqam')

def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def corr(q,g):
    return {'pearson':float(pearsonr(q,g).statistic),'spearman':float(spearmanr(q,g).statistic)} if np.std(q)>1e-10 and np.std(g)>1e-10 else {'pearson':None,'spearman':None}

def select():
    records=[]
    for directory in ['pilot_libero_v1','dev_libero_v1']:
        records+=json.loads((ROOT/'data'/directory/'manifest.json').read_text())['records']
    rng=np.random.default_rng(20261007)
    selected=[]
    for outcome in [True,False]:
        subset=[r for r in records if bool(r['success'])==outcome]
        selected += [subset[int(i)] for i in rng.choice(len(subset),10,replace=False)]
    episodes=[]
    for row in selected:
        with np.load(row['path'],allow_pickle=False) as f:
            episodes.append((row,{k:f[k] for k in f.files if k not in ['model_xml','snapshot_metadata']}))
    return episodes

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);args=p.parse_args()
    episodes=select();enc_cfg=encoder_config(ROOT);tokenizer=DIVLTokenizer(enc_cfg)
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    normal=transforms.Normalize(norm,use_quantiles=True)
    samples=[(e,i) for _,e in episodes for i in range(len(e['rewards']))]
    rows=[];errors={'action_pairing':0.,'chunk_reward':0.,'discount':0.,'mc_return':0.,'terminal_bootstrap':0.,'current_image':0.,'next_image':0.}
    for manifest,e in episodes:
        offset=0
        mc=np.asarray([sum(.9999**j*float(v) for j,v in enumerate(e['env_step_rewards'][int(e['lengths'][:i].sum()):])) for i in range(len(e['rewards']))])
        for i,kvalue in enumerate(e['lengths']):
            k=int(kvalue);done=bool(e['terminal'][i]);native=e['env_step_rewards'][offset:offset+k]
            reward=sum(.9999**j*float(v) for j,v in enumerate(native))
            action=e['actions'][i][:k];original=e['raw_env_actions'][offset:offset+k]
            errors['action_pairing']=max(errors['action_pairing'],float(np.max(np.abs(action-original))))
            errors['chunk_reward']=max(errors['chunk_reward'],abs(float(e['rewards'][i])-reward))
            errors['discount']=max(errors['discount'],abs(float(e['discounts'][i])-(0. if done else .9999**k)))
            errors['mc_return']=max(errors['mc_return'],abs(float(e['mc_returns'][i])-float(mc[i])))
            if done:errors['terminal_bootstrap']=max(errors['terminal_bootstrap'],abs(float(e['discounts'][i])))
            rows.append({'episode_id':manifest['episode_id'],'split':manifest['split'],'task':manifest['task_id'],
                         'chunk_index':i,'env_step':offset,'state_id':manifest['episode_id']+f'_chunk{i:03d}',
                         'state_sha256':sha(e['state'][i]),'next_state_sha256':sha(e['state'][i+1]),
                         'action':json.dumps(action.tolist()),'k':k,'reward_chunk':float(e['rewards'][i]),
                         'discount':float(e['discounts'][i]),'done':done,'success':bool(e['success']),
                         'mc_future_return':float(mc[i])})
            offset+=k
        assert offset==len(e['env_step_rewards'])
    per_method={}
    for method in ['scalar','divl']:
        critic=build_critic(LiberoTrainingConfig(value_type=method),enc_cfg)
        cp=ROOT/'checkpoints/critic_pilot_v1'/method/'step2000'
        with ocp.PyTreeCheckpointer() as ck: restored=ck.restore(cp.resolve())
        target=nnx.clone(critic)
        for model,key in [(critic,'critic'),(target,'target')]:
            state=nnx.state(model,nnx.Param);state.replace_by_pure_dict(restored[key]);nnx.update(model,state);model.eval()
        del restored
        @nnx.jit
        def evaluate(model,targ,batch):
            z=model.encoder(**batch['obs'],train=False)
            nz=targ.encoder(**batch['next_obs'],train=False)
            q=jnp.minimum(*model.q_head(z,batch['actions'],batch['action_mask']))
            v=model.v_head(z);nv=targ.v_head(nz)
            if method=='divl':
                v=categorical_quantile(v,model.v_head.atoms,adaptive_quantile_level(v,tau_base=.6,alpha=.3,tau_min=0,tau_max=1))
                nv=categorical_quantile(nv,targ.v_head.atoms,adaptive_quantile_level(nv,tau_base=.6,alpha=.3,tau_min=0,tau_max=1))
            td=batch['rewards']+batch['discounts']*nv
            return q,v,nv,td
        predictions=[]
        for start in range(0,len(samples),args.batch):
            subset=samples[start:start+args.batch]
            batch=make_batch(subset,tokenizer,norm,td_steps=1)
            expected=normal({'actions':np.stack([e['actions'][i] for e,i in subset])})['actions']
            expected=np.where(np.stack([e['action_mask'][i] for e,i in subset])[...,None],expected,0)
            np.testing.assert_allclose(np.asarray(batch['actions']),expected,atol=1e-7)
            np.testing.assert_allclose(np.asarray(batch['rewards']),[e['rewards'][i] for e,i in subset],atol=1e-7)
            np.testing.assert_allclose(np.asarray(batch['discounts']),[e['discounts'][i] for e,i in subset],atol=1e-7)
            for key,field in [('base_0_rgb','base'),('left_wrist_0_rgb','wrist')]:
                cur=np.stack([e[field][i] for e,i in subset]).astype(np.float32)/127.5-1.
                nxt=np.stack([e[field][i+1] for e,i in subset]).astype(np.float32)/127.5-1.
                errors['current_image']=max(errors['current_image'],float(np.max(np.abs(np.asarray(batch['obs']['images'][key])-cur))))
                errors['next_image']=max(errors['next_image'],float(np.max(np.abs(np.asarray(batch['next_obs']['images'][key])-nxt))))
            pred=np.stack([np.asarray(x) for x in evaluate(critic,target,batch)],axis=-1)
            predictions.extend(pred.tolist())
            print('Q_ALIGNMENT_PROGRESS',method,start+len(subset),'/',len(samples),flush=True)
        for row,pred in zip(rows,predictions,strict=True):
            for name,value in zip(['q','value','target_next_value','td_target'],pred,strict=True):row[method+'_'+name]=float(value)
        q=np.asarray([d[0] for d in predictions]);g=np.asarray([row['mc_future_return'] for row in rows])
        per_method[method]={'all_chunks':corr(q,g),'success_only':corr(q[g>0],g[g>0]),
                            'success_mean_q':float(q[g>0].mean()),'failure_mean_q':float(q[g==0].mean()),
                            'per_task':{str(t):corr(q[np.asarray([row['task']==t for row in rows])],g[np.asarray([row['task']==t for row in rows])]) for t in sorted(set(row['task'] for row in rows))}}
        del critic,target;jax.clear_caches()
    passed=errors['action_pairing']==0 and errors['terminal_bootstrap']==0 and errors['current_image']==0 and errors['next_image']==0 and max(errors['discount'],errors['chunk_reward'],errors['mc_return'])<1e-6
    csvpath=ROOT/'artifacts/q_alignment_chunks.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotdir=ROOT/'reports/q_alignment_plots';plotdir.mkdir(parents=True,exist_ok=True)
    for episode,_ in episodes:
        subset=[v for v in rows if v['episode_id']==episode['episode_id']]
        fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True)
        timeaxis=[v['env_step'] for v in subset]
        for method in ['scalar','divl']:axes[0].plot(timeaxis,[v[method+'_q'] for v in subset],label=method)
        axes[0].legend();axes[0].set_ylabel('Q')
        axes[1].plot(timeaxis,[v['mc_future_return'] for v in subset]);axes[1].set_ylabel('MC return')
        axes[2].plot(timeaxis,[v['reward_chunk'] for v in subset]);axes[2].set_ylabel('reward')
        axes[2].set_xlabel('Environment step');axes[2].axvline(timeaxis[-1],color='red',linestyle='--',label='terminal chunk');axes[2].legend()
        fig.suptitle(episode['episode_id']);fig.tight_layout();fig.savefig(plotdir/(episode['episode_id']+'.png'));plt.close(fig)
    result={'alignment_passed':passed,'target_alignment_bug_found':not passed,'max_errors':errors,
            'episodes':[v for v,_ in episodes],'success_episodes':10,'failure_episodes':10,'chunks':len(rows),'methods':per_method,
            'source_implementation':'make_batch td_steps=1; Q(s_i,A_i), V target(s_i+1), native discounted reward chunk, gamma^actual_k or zero terminal',
            'no_training_or_actor_update':True,'csv':str(csvpath),'plots':str(plotdir),
            'scope':'Random fixed-seed sample from existing train+dev; no final split. Correlations are diagnostic, not held-out policy performance.'}
    (ROOT/'artifacts/q_alignment_audit.json').write_text(json.dumps(result,indent=2))
    (ROOT/'reports/q_return_alignment_audit.md').write_text('# Q-return alignment audit\n\n'+json.dumps(result,indent=2)+'\n\nNegative global Pearson may be driven by task/outcome groups and bootstrapped Q calibration; within-task/outcome results above must be inspected. It does not demonstrate action ranking.\n')
    print('Q_ALIGNMENT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
