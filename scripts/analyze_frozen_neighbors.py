# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Observational local-future analysis with unchanged public SigLIP, not learned Q embeddings."""
from pathlib import Path
import json
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image
from scipy.spatial.distance import cdist
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.models.divl_losses import adaptive_quantile_level
from openpi.shared import normalize
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config

ROOT=Path('/opt/divl-trqam')

def main():
    manifest=json.loads((ROOT/'data/pilot_libero_v1/manifest.json').read_text())
    samples=[];metadata=[]
    for row in manifest['records']:
        with np.load(row['path'],allow_pickle=False) as f:ep={k:f[k] for k in f.files}
        for fraction in [.35,.70]:
            i=min(len(ep['rewards'])-1,int(len(ep['rewards'])*fraction));samples.append((ep,i));metadata.append({'episode':row['episode_id'],'task':row['task_id'],'index':i,'return':float(ep['mc_returns'][i])})
    cfg=encoder_config(ROOT);critic=build_critic(LiberoTrainingConfig(value_type='divl'),cfg)
    @nnx.jit
    def frozen_features(vision,base,wrist):
        a,_=vision.siglip(base,train=False);b,_=vision.siglip(wrist,train=False)
        return jnp.concatenate([a.mean(axis=1),b.mean(axis=1)],axis=-1)
    embeddings=[]
    for start in range(0,len(samples),8):
        batch=samples[start:start+8]
        base=np.stack([e['base'][i] for e,i in batch]).astype(np.float32)/127.5-1
        wrist=np.stack([e['wrist'][i] for e,i in batch]).astype(np.float32)/127.5-1
        embeddings.extend(np.asarray(frozen_features(critic.encoder.vision,jnp.asarray(base),jnp.asarray(wrist))))
    embeddings=np.stack(embeddings);embeddings/=np.maximum(np.linalg.norm(embeddings,axis=-1,keepdims=True),1e-10)
    np.savez_compressed(ROOT/'artifacts/frozen_neighbor_features_v2.npz',embeddings=embeddings,metadata=np.asarray(json.dumps(metadata)),encoder_checkpoint=np.asarray(str(cfg.siglip_checkpoint)))
    print('FROZEN_FEATURES_COMPLETE',len(samples),'dimensions',embeddings.shape[-1],'devices',[str(v) for v in jax.devices()],flush=True)
    candidates=[];unconditional=[]
    for task in range(10):
        ids=[i for i,m in enumerate(metadata) if m['task']==task]
        distance=cdist(embeddings[ids],embeddings[ids],metric='cosine')
        for a,ia in enumerate(ids):
            all_other=[b for b,ib in enumerate(ids) if metadata[ia]['episode']!=metadata[ib]['episode']]
            nearest=sorted(all_other,key=lambda b:distance[a,b])[:5]
            futures=np.asarray([metadata[ids[b]]['return'] for b in nearest])
            unconditional.append({'state_index':ia,'task':task,'neighbor_indices':[ids[b] for b in nearest],
              'cosine_distances':[float(distance[a,b]) for b in nearest],'future_returns':futures.tolist(),
              'future_variance':float(futures.var()),'has_success_failure_mixture':bool(np.any(futures>0) and np.any(futures==0))})
            possible=[b for b,ib in enumerate(ids) if metadata[ia]['episode']!=metadata[ib]['episode'] and abs(metadata[ia]['return']-metadata[ib]['return'])>.1]
            if not possible:continue
            b=min(possible,key=lambda b:distance[a,b]);ib=ids[b]
            e,i=samples[ia];f,j=samples[ib]
            candidates.append({'left':ia,'right':ib,'task':task,'cosine_distance':float(distance[a,b]),'proprio_distance':float(np.linalg.norm(e['state'][i]-f['state'][j])),
                'environment_action_distance':float(np.linalg.norm(e['actions'][i]-f['actions'][j])),'future_left':metadata[ia]['return'],'future_right':metadata[ib]['return']})
    # Reporting examples chosen by embedding distance, no model performance selection.
    chosen=[];seen=set()
    for row in sorted(candidates,key=lambda c:c['cosine_distance']):
        pair=tuple(sorted([row['left'],row['right']]))
        if pair in seen:continue
        seen.add(pair);chosen.append(row)
        if len(chosen)==20:break
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero');tokenizer=DIVLTokenizer(cfg)
    ids=sorted({c[k] for c in chosen for k in ['left','right']});outputs={}
    @nnx.jit
    def predict(model,batch):
        z=model.encoder(**batch['obs'],train=False);q=jnp.minimum(*model.q_head(z,batch['actions'],batch['action_mask']));v=model.v_head(z)
        return q,v
    for method in ['scalar','divl']:
        if method=='scalar':critic=build_critic(LiberoTrainingConfig(value_type=method),cfg)
        else:critic=build_critic(LiberoTrainingConfig(value_type=method),cfg)
        with ocp.PyTreeCheckpointer() as ck:r=ck.restore((ROOT/'checkpoints/critic_pilot_v1'/method/'step2000').resolve())
        state=nnx.state(critic,nnx.Param);state.replace_by_pure_dict(r['critic']);nnx.update(critic,state);critic.eval()
        del r
        for index in ids:
            q,v=predict(critic,make_batch([samples[index]],tokenizer,norm,td_steps=1));out={'q':float(q[0])}
            if method=='divl':
                logits=v[0];p=np.asarray(jax.nn.softmax(logits.astype(jnp.float32)));atoms=np.asarray(critic.v_head.atoms)
                tau=float(adaptive_quantile_level(logits[None],tau_base=.6,alpha=.3,tau_min=0,tau_max=1)[0])
                out.update(entropy=float(-np.sum(p*np.log(np.maximum(p,1e-10)))),tau=tau,quantile=float(atoms[min(int(np.searchsorted(p.cumsum(),tau)),200)]),distribution=p.tolist())
            outputs.setdefault(str(index),{})[method]=out
    images=ROOT/'reports/frozen_neighbor_examples';images.mkdir(parents=True,exist_ok=True)
    details='\n## Actual local examples\n\n'
    table='| Task | Embedding distance | Proprio distance | Action distance | Future returns | Scalar Q | DIVL Q |\n|---|---:|---:|---:|---|---|---|\n'
    for n,c in enumerate(chosen):
        a,b=c['left'],c['right'];left=samples[a];right=samples[b]
        pair=np.concatenate([np.concatenate([left[0]['base'][left[1]],left[0]['wrist'][left[1]]],axis=1),
                             np.concatenate([right[0]['base'][right[1]],right[0]['wrist'][right[1]]],axis=1)],axis=0)
        Image.fromarray(pair).save(images/f'example{n:02d}.png')
        table+=f'| {c["task"]} | {c["cosine_distance"]:.5f} | {c["proprio_distance"]:.4f} | {c["environment_action_distance"]:.4f} | {c["future_left"]:.4f}, {c["future_right"]:.4f} | {outputs[str(a)]["scalar"]["q"]:.4f}, {outputs[str(b)]["scalar"]["q"]:.4f} | {outputs[str(a)]["divl"]["q"]:.4f}, {outputs[str(b)]["divl"]["q"]:.4f} |\n'
        details+=f'### Example {n+1}, task{c["task"]}\n\n![Base/wrist, left state above right state](frozen_neighbor_examples/example{n:02d}.png)\n\n'
        details+='| State | True future return | Scalar Q | DIVL Q | DIVL quantile | Entropy | Tau |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for label,index,value in [('left',a,c['future_left']),('right',b,c['future_right'])]:
            out=outputs[str(index)];v=out['divl']
            details+=f'| {label} | {value:.5f} | {out["scalar"]["q"]:.5f} | {v["q"]:.5f} | {v["quantile"]:.5f} | {v["entropy"]:.5f} | {v["tau"]:.5f} |\n'
    result={'encoder':'Public frozen pretrained SigLIP before any critic updates; mean patch features from two cameras',
        'encoder_provenance':{'checkpoint':str(cfg.siglip_checkpoint),'loader':'build_critic -> load_siglip_weights; features extracted before any learned critic checkpoint is loaded','updates':0},
        'samples':len(samples),'candidate_pairs':len(candidates),'examples':chosen,'predictions':outputs,
        'unconditional_nearest5':{'different_episode_same_task':True,'states':len(unconditional),
            'success_failure_mixture_ratio':float(np.mean([r['has_success_failure_mixture'] for r in unconditional])),
            'mean_future_variance':float(np.mean([r['future_variance'] for r in unconditional])),
            'records':unconditional},
        'example_selection':'20 unique closest unordered pairs satisfying abs(future difference)>.1. This deliberately conditioned subset does not estimate prevalence; unconditional nearest5 metrics are separate.',
        'limitation':'Observational feature neighbors do not establish equal full state or a causal P(G|s,a). Same-snapshot counterfactual experiments provide separate action evidence.'}
    (ROOT/'artifacts/frozen_state_future_neighbors.json').write_text(json.dumps(result,indent=2))
    summary={k:v for k,v in result['unconditional_nearest5'].items() if k!='records'}
    (ROOT/'reports/similar_state_future_analysis.md').write_text('# Frozen-embedding state / future analysis\n\n'+result['encoder']+'\n\n'+result['example_selection']+'\n\nUnconditional same-task/different-episode nearest5:\n\n'+json.dumps(summary,indent=2)+'\n\n'+table+'\n'+result['limitation']+'\n'+details)
    print(json.dumps({k:v for k,v in result.items() if k not in ['examples','predictions','unconditional_nearest5']},indent=2),flush=True)

if __name__=='__main__':main()
