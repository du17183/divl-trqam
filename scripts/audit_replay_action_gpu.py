# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Read-only public feature extraction and retained-critic conditioning probes."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import build_critic, encoder_config, LiberoTrainingConfig

R=Path('/opt/divl-trqam')
OUT=R/'artifacts/replay_action_audit_v3'
FIELDS=['base','wrist','state','actions','action_mask','rewards','discounts','mc_returns','lengths','success','prompt','raw_env_actions']

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in FIELDS if k in z.files}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    low=np.asarray(norm['actions'].q01)[:7];high=np.asarray(norm['actions'].q99)[:7]
    def normal(a,m):return np.where(np.asarray(m)[...,None],2*(a-low)/(high-low+1e-6)-1,0).astype(np.float32)
    manifest=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())
    images=[];wrists=[];actions=[];proprio=[];meta=[];masks=[]
    for n,row in enumerate(manifest['records']):
        e=load(row['path'])
        for fraction in [.1,.3,.5,.7,.9]:
            i=min(len(e['rewards'])-1,int(len(e['rewards'])*fraction))
            images.append(e['base'][i].copy());wrists.append(e['wrist'][i].copy());proprio.append(e['state'][i].copy())
            masks.append(e['action_mask'][i]);actions.append(normal(e['actions'][i],e['action_mask'][i]))
            meta.append(dict(episode_id=row['episode_id'],task_id=row['task_id'],initial_state_id=row['initial_state_id'],index=i,
                success=row['success'],future_return=float(e['mc_returns'][i]),remaining_steps=int(e['lengths'][i:].sum()),k=int(e['lengths'][i]),path=row['path']))
        if n%25==0:print('LOAD',n,flush=True)
    selected=sorted(json.loads((R/'artifacts/counterfactual_results_v2.json').read_text()),key=lambda x:x['metadata']['state_id'])
    dev=[];devsamples=[]
    for row in selected:
        m=row['metadata'];e=load(m['episode_path']);i=m['index'];devsamples.append((e,i));dev.append(m)
        images.append(e['base'][i].copy());wrists.append(e['wrist'][i].copy())
    cfg=encoder_config(R);model=build_critic(LiberoTrainingConfig(value_type='scalar'),cfg)
    @nnx.jit
    def feature(v,a,b):
        x,_=v.siglip(a,train=False);y,_=v.siglip(b,train=False)
        return jnp.concatenate([x.mean(1),y.mean(1)],axis=-1).astype(jnp.float32)
    feats=[]
    for i in range(0,len(images),8):
        a=np.stack(images[i:i+8]).astype(np.float32)/127.5-1;b=np.stack(wrists[i:i+8]).astype(np.float32)/127.5-1
        feats.extend(np.asarray(feature(model.encoder.vision,jnp.asarray(a),jnp.asarray(b))))
        if i%200==0:print('PUBLIC_FEATURES',i,len(images),flush=True)
    feats=np.stack(feats);feats/=np.maximum(np.linalg.norm(feats,axis=1,keepdims=True),1e-10)
    np.savez_compressed(OUT/'public_features.npz',train_features=feats[:len(meta)],dev_features=feats[len(meta):],
        train_actions=np.stack(actions),train_masks=np.stack(masks),train_proprio=np.stack(proprio),
        train_metadata=np.asarray(json.dumps(meta)),dev_metadata=np.asarray(json.dumps(dev)),
        encoder_checkpoint=np.asarray(str(cfg.siglip_checkpoint)),action_low=low,action_high=high)
    del images,wrists,feats
    tokenizer=DIVLTokenizer(cfg);rng=np.random.default_rng(2026100717)
    valid_train=np.stack(actions)[np.stack(masks)]
    arm_mean=valid_train.mean(axis=0)
    std=valid_train.std(axis=0)
    shared_perm=rng.permutation(50)
    shared_direction=rng.normal(size=(50,10,7)).astype(np.float32)
    @nnx.jit
    def encode(c,obs):return c.encoder(**obs,train=False)
    @nnx.jit
    def probe(h,z,a,mask):
        def fun(z,a):return jnp.minimum(*h(z,a,mask)).sum()
        gz,ga=jax.grad(fun,argnums=(0,1))(z,a)
        pool=h.action_pool(a,mask);tokens=jnp.tanh(h.action_pool.action_proj(a)+h.action_pool.position_embedding.value[None,:a.shape[1]])
        f=jnp.concatenate([z.astype(jnp.float32),pool],-1)
        layer=h.q1.net.layers[0]
        state_in=z.astype(jnp.float32)@layer.kernel.value[:z.shape[1]]
        action_in=pool@layer.kernel.value[z.shape[1]:]
        return jnp.minimum(*h(z,a,mask)),gz,ga,pool,tokens,state_in,action_in
    @nnx.jit
    def qs(h,z,a,mask):return jnp.minimum(*h(z,a,mask))
    results={}
    for method in ['scalar','divl']:
        if method=='divl':model=build_critic(LiberoTrainingConfig(value_type='divl'),cfg)
        with ocp.PyTreeCheckpointer() as ck:saved=ck.restore((R/'checkpoints/critic_pilot_v1'/method/'step2000').resolve())
        state=nnx.state(model,nnx.Param);state.replace_by_pure_dict(saved['critic']);nnx.update(model,state);model.eval();del saved
        batch=make_batch(devsamples,tokenizer,norm,td_steps=1)
        zs=[]
        for i in range(0,50,5):zs.append(np.asarray(encode(model,jax.tree.map(lambda x:x[i:i+5],batch['obs']))))
        z=jnp.asarray(np.concatenate(zs));a=batch['actions'];mask=batch['action_mask']
        q,gz,ga,pool,tokens,si,ai=probe(model.q_head,z,a,mask)
        q=np.asarray(q);perm=shared_perm;mean=np.broadcast_to(arm_mean,(50,10,7)).copy()
        original=np.asarray(a);means=np.where(np.asarray(mask)[...,None],mean,0)
        direction=shared_direction
        variants={'behavior':original,'batch_shuffle':original[perm],'zero':np.zeros_like(original),'replay_mean':means,
            'arm_only':np.concatenate([original[:,:,:6],means[:,:,6:]],-1),
            'gripper_only':np.concatenate([means[:,:,:6],original[:,:,6:]],-1),
            'small_0.25std':original+.25*direction*std,'medium_0.5std':original+.5*direction*std}
        record={}
        qscale=max(float(q.std()),1e-10)
        for name,value in variants.items():
            physical=np.clip((value+1)/2*(high-low+1e-6)+low,-1,1)
            value=normal(physical,np.asarray(mask)) if name not in ['zero','replay_mean','arm_only','gripper_only','batch_shuffle','behavior'] else np.where(np.asarray(mask)[...,None],value,0)
            v=np.asarray(qs(model.q_head,z,jnp.asarray(value),mask));d=v-q
            record[name]=dict(mean_delta=float(d.mean()),mean_abs_delta=float(abs(d).mean()),median_abs_delta=float(np.median(abs(d))),relative_to_state_Q_std=float(abs(d).mean()/qscale),q=v.tolist())
        normaxis=lambda x:np.linalg.norm(np.asarray(x).astype(np.float32),axis=-1)
        gn=np.linalg.norm(np.asarray(ga).reshape(50,-1),axis=1);zn=normaxis(gz)
        results[method]=dict(ablations=record,q_std=float(q.std()),action_gradient_norm=gn.tolist(),state_feature_gradient_norm=zn.tolist(),
            gradient_ratio=(gn/np.maximum(zn,1e-10)).tolist(),state_norm=normaxis(z).tolist(),action_feature_norm=normaxis(pool).tolist(),
            state_rms=np.sqrt(np.mean(np.asarray(z,dtype=np.float32)**2,axis=1)).tolist(),
            action_rms=np.sqrt(np.mean(np.asarray(pool)**2,axis=1)).tolist(),
            first_layer_state_contribution_norm=normaxis(si).tolist(),first_layer_action_contribution_norm=normaxis(ai).tolist(),
            action_token_norm=np.mean(normaxis(tokens),axis=1).tolist(),unused_gradient_max=float(np.max(np.abs(np.asarray(ga)[~np.asarray(mask)]))))
        np.savez_compressed(OUT/f'{method}_conditioning.npz',z=np.asarray(z,dtype=np.float32),actions=original,mask=np.asarray(mask),gradient=np.asarray(ga),state_gradient=np.asarray(gz),pool=np.asarray(pool))
        (OUT/'conditioning.json').write_text(json.dumps(results,indent=2));print('CONDITIONING',method,json.dumps({k:float(np.mean(v)) for k,v in results[method].items() if isinstance(v,list)}),flush=True)
    print('GPU_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
