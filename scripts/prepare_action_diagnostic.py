# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Held-out same-state candidates; compare all critic budgets without actor updates."""
import argparse
import json
import hashlib
from pathlib import Path
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from scipy.stats import spearmanr,pearsonr
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config

ROOT=Path('/opt/divl-trqam')

def load_episode(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def select_states():
    manifest=json.loads((ROOT/'data/dev_libero_v1/manifest.json').read_text())
    selected=[]
    for task in range(10):
        episodes=sorted([r for r in manifest['records'] if r['task_id']==task],key=lambda r:r['episode_id'])
        for order,row in enumerate(episodes):
            ep=load_episode(row['path'])
            # One state/initial ID, outcome-independent stage coverage, deterministic.
            index=min(len(ep['rewards'])-1,int(len(ep['rewards'])*[.2,.4,.6,.8][order%4]))
            selected.append((row,ep,index))
    assert len(selected)==100
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--port',type=int,default=8027);args=p.parse_args()
    policy=WebsocketClientPolicy('127.0.0.1',args.port)
    manifest=json.loads((ROOT/'data/pilot_libero_v1/manifest.json').read_text())
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    low=np.asarray(norm['actions'].q01)[:7];high=np.asarray(norm['actions'].q99)[:7]
    def normal(a):return (2*(a-low)/(high-low+1e-6)-1).astype(np.float32)
    def raw(a):return np.clip((a+1)/2*(high-low+1e-6)+low,-1,1).astype(np.float32)
    enc_cfg=encoder_config(ROOT);tokenizer=DIVLTokenizer(enc_cfg)
    critic=build_critic(LiberoTrainingConfig(value_type=args.method),enc_cfg)
    cp=ROOT/'checkpoints/critic_pilot_v1'/args.method/f'step{args.steps:04d}'
    with ocp.PyTreeCheckpointer() as ck:restored=ck.restore(cp.resolve())
    state=nnx.state(critic,nnx.Param);state.replace_by_pure_dict(restored['critic']);nnx.update(critic,state);critic.eval()
    @nnx.jit
    def encode(model,obs):return model.encoder(**obs,train=False)
    @nnx.jit
    def values(head,z,actions,mask):return jnp.minimum(*head(z,actions,mask))
    @nnx.jit
    def gradient(head,z,a,mask):
        return nnx.grad(lambda h,x:jnp.minimum(*h(z,x,mask)).sum(),argnums=1)(head,a)
    rng=np.random.default_rng(20261007);selected=select_states();records=[];q_all=[];original=[];shuffled=[];gradnorm=[]
    destination=ROOT/'data/action_diagnostic_v1';destination.mkdir(parents=True,exist_ok=True)
    train_by_task={}
    for task in range(10):
        rows=[r for r in manifest['records'] if r['task_id']==task]
        train_by_task[task]=[load_episode(r['path']) for r in rows]
    for order,(row,ep,index) in enumerate(selected):
        task=row['task_id'];mask=np.asarray(ep['action_mask'][index]);k=int(mask.sum())
        candidates=[]
        behavior=np.asarray(ep['actions'][index]);candidates.append(('behavior',behavior.copy()))
        plan=policy.infer({'observation/image':ep['base'][index],'observation/wrist_image':ep['wrist'][index],
            'observation/state':ep['state'][index],'prompt':str(ep['prompt']),'_noise_seed':2026100900+order})['actions']
        actor=np.zeros((10,7),np.float32);actor[:k]=np.clip(plan[:k],-1,1);candidates.append(('frozen_actor_fresh_noise',actor))
        nearest=min(((float(np.linalg.norm(e['state'][i]-ep['state'][index])),e,i) for e in train_by_task[task] for i in range(len(e['rewards']))),key=lambda t:t[0])
        neighbor=np.zeros((10,7),np.float32);neighbor[:k]=nearest[1]['actor_bc_executed_sequence'][nearest[2]][:k];candidates.append(('nearby_replay',np.clip(neighbor,-1,1)))
        base=np.where(mask[:,None],normal(behavior),0)
        direction=rng.normal(size=(10,7)).astype(np.float32)*mask[:,None];direction/=np.linalg.norm(direction)
        for magnitude in [.03,.10]:
            for sign in [-1,1]:
                a=raw(base+sign*magnitude*direction);a[~mask]=0
                candidates.append((f'perturb_{magnitude}_{sign}',a))
        gripper=behavior.copy();gripper[:k,-1]*=-1;candidates.append(('gripper_flip',gripper));assert len(candidates)==8
        batch=make_batch([(ep,index)],tokenizer,norm,td_steps=1);z=encode(critic,batch['obs'])
        actions=np.stack([np.where(mask[:,None],normal(a),0) for _,a in candidates])
        q=np.asarray(values(critic.q_head,jnp.repeat(z,8,axis=0),jnp.asarray(actions),jnp.tile(mask,(8,1))))
        g=np.asarray(gradient(critic.q_head,z,jnp.asarray(base[None]),jnp.asarray(mask[None])))[0].copy()
        g*=mask[:,None];gn=float(np.linalg.norm(g));gradnorm.append(gn)
        if not np.isfinite(gn) or not np.isfinite(q).all():raise FloatingPointError('Nonfinite diagnostic')
        metadata={'state_id':row['episode_id']+f'_chunk{index:03d}','episode_path':row['path'],'task_id':task,'initial_state_id':row['initial_state_id'],
                  'index':index,'seed':row['seed'],'k':k,'remaining_steps':520-int(ep['lengths'][:index].sum()),'future_return':float(ep['mc_returns'][index]),
                  'candidate_names':[n for n,a in candidates],'nearby_proprio_distance':nearest[0],'gradient_norm':gn,
                  'selected_for_branches':order%10<5,'selection':'10 states/task, one per dev initial ID; first five/task used for branches; no outcome selection'}
        arrays={'candidate_actions':np.stack([a for n,a in candidates]),'candidate_q':q,'action_mask':mask,'gradient':g,'proprio':ep['state'][index]}
        for eps in [.01,.03,.05]:
            for sign in [-1,1]:
                a=raw(base+sign*eps*g/max(gn,1e-12));a[~mask]=0
                arrays[f'gradient_{eps}_{sign}']=a
        outfile=destination/f'{metadata["state_id"]}_{args.method}_step{args.steps}.npz'
        np.savez_compressed(outfile,metadata=np.asarray(json.dumps(metadata)),**arrays)
        records.append({**metadata,'path':str(outfile),'q':q.tolist()});q_all.append(q);original.append(q[0])
        print('STATIC_STATE '+json.dumps({'method':args.method,'completed':order+1,'q_variance':float(q.var())}),flush=True)
    # Genuine batch action shuffle, preserving each recipient state's executed mask.
    permutation=np.random.default_rng(20261008).permutation(100)
    for order,(_,ep,index) in enumerate(selected):
        other=selected[permutation[order]][1];oi=selected[permutation[order]][2]
        mask=ep['action_mask'][index];k=int(mask.sum())
        a=np.zeros((10,7),np.float32);a[:k]=other['actor_bc_executed_sequence'][oi][:k]
        batch=make_batch([(ep,index)],tokenizer,norm,td_steps=1);z=encode(critic,batch['obs'])
        normalized=np.where(mask[:,None],normal(a),0)
        shuffled.append(float(values(critic.q_head,z,jnp.asarray(normalized[None]),jnp.asarray(mask[None]))[0]))
    qs=np.stack(q_all);var_total=float(qs.var());rho=float(qs.var(axis=1).mean()/max(var_total,1e-12))
    returns=np.asarray([r['future_return'] for r in records]);orig=np.asarray(original)
    result={'method':args.method,'steps':args.steps,'states':100,'candidate_count':8,'rho':rho,'mean_action_variance':float(qs.var(axis=1).mean()),
            'total_q_variance':var_total,'shuffle_mean_absolute_delta':float(np.abs(orig-shuffled).mean()),
            'gradient_norm_mean':float(np.mean(gradnorm)),'return_pearson':float(pearsonr(orig,returns).statistic) if returns.std()>0 and orig.std()>1e-10 else None,
            'return_spearman':float(spearmanr(orig,returns).statistic) if returns.std()>0 and orig.std()>1e-10 else None,
            'scope':'Static action sensitivity; not proof of true action quality or QAM gradient correctness','records':records}
    result['diagnostic_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['checkpoint_metrics_sha256']=hashlib.sha256((cp.parent/f'metrics_step{args.steps:04d}.json').read_bytes()).hexdigest()
    result['output_hashes']={r['state_id']:hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in records}
    (ROOT/f'artifacts/static_action_{args.method}_step{args.steps}.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
