# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Matched critic-only repair: local intervention replay + original LIBERO n-step setting."""
from pathlib import Path
import argparse,json,hashlib,time,dataclasses,copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import flax.nnx as nnx
import jax,jax.numpy as jnp
import orbax.checkpoint as ocp
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config,initialize_training,critic_step
from train_libero_lwd import digest,CRITIC_REPLAY_FIELDS
from openpi.training.precision import assert_fp32_master
from critic_data_parallel import CriticDataParallel

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3';D=R/'data/action_sensitive_replay_v2'

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True)
    p.add_argument('--device-count',type=int,default=1);p.add_argument('--loader-workers',type=int,default=8);args=p.parse_args()
    assert jax.device_count()==args.device_count and 256%args.device_count==0
    data=json.loads((D/'manifest.json').read_text());gate=json.loads((O/'replay_v2_data_gate.json').read_text());assert gate['passed']
    assert data['states']==1000 and data['outcomes']==5000 and not data['dev_states_used']
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text());assert data['split_hash']==split['sha256']
    original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text());rows=original['records']+data['records'];episodes=[];identities=[];orig_indices=[];anchors=[];cont=[]
    def load_episode(row):
        path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with np.load(path,allow_pickle=False) as f:e={k:f[k] for k in CRITIC_REPLAY_FIELDS}
        assert int(e['init_id']) in split['tasks'][str(e['task_name'])]['train']
        return e
    # map yields manifest order even though decoding runs concurrently.
    loader=ThreadPoolExecutor(max_workers=args.loader_workers)
    for n,(row,e) in enumerate(zip(rows,loader.map(load_episode,rows),strict=True)):
        episodes.append(e);identities.append(row['episode_id'])
        if n<len(original['records']):orig_indices.extend((n,i) for i in range(len(e['rewards'])))
        else:anchors.append((n,0));cont.extend((n,i) for i in range(1,len(e['rewards'])))
        if n%500==0:print('REPLAY_LOAD',n,len(rows),flush=True)
    loader.shutdown()
    orig_indices=np.asarray(orig_indices);anchors=np.asarray(anchors);cont=np.asarray(cont)
    cfg=LiberoTrainingConfig(value_type=args.method,reference_type='fixed',actor_updates=False)
    enc_cfg=encoder_config(R);tokenizer=DIVLTokenizer(enc_cfg);norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    critic=build_critic(cfg,enc_cfg);target,vopt,qopt=initialize_training(critic,2000,learning_rate=.0002)
    assert_fp32_master(critic);assert_fp32_master(target)
    initial=dict(q_initial_hash=digest(critic.q_head),encoder_initial_hash=digest(critic.encoder))
    dp=CriticDataParallel(critic,target,vopt,qopt,jax.devices())
    rng=np.random.default_rng(cfg.seed);orderhash=hashlib.sha256();metrics=[];started=time.monotonic();out=R/'checkpoints/critic_replay_v2'/args.method
    assert not out.exists(),'Fresh paired experiment must not overwrite existing checkpoints';out.mkdir(parents=True)
    protocol=dict(config=dataclasses.asdict(cfg),batch_size=256,td_chunks=10,steps=2000,seed=cfg.seed,learning_rate=.0002,
        execution=dict(data_parallel_devices=args.device_count,global_batch=256,microbatch_per_device=256//args.device_count,
                       gradient_reduction='GSPMD global mean / all-reduce',loader_workers=args.loader_workers,prefetch_batches=1,
                       checkpoint_array_storage='Host NumPy, unchanged FP32 values; portable across mesh sizes'),
        replay_manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),sampling='128 uniform branch anchors +64 uniform original chunks +64 uniform branch continuation chunks, shuffled each batch; all outcomes retained',
        note='Shared replay repair and restoration of original LIBERO adapter td10; no actor/QAM/DIVL formulation change. Two simultaneous shared repairs cannot be causally disentangled by this comparison.',**initial)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def draw():
        index=np.concatenate([anchors[rng.integers(len(anchors),size=128)],orig_indices[rng.integers(len(orig_indices),size=64)],cont[rng.integers(len(cont),size=64)]])
        index=index[rng.permutation(256)];samples=[(episodes[e],int(i)) for e,i in index]
        return samples,json.dumps([(identities[e],int(i)) for e,i in index],separators=(',',':')).encode(),copy.deepcopy(rng.bit_generator.state)
    def prepare(samples):
        batch=make_batch(samples,tokenizer,norm,td_steps=10,return_numpy=True)
        batch['critic_k']=np.asarray([int(e['lengths'][i]) for e,i in samples],dtype=np.int32)
        return batch
    prefetch=ThreadPoolExecutor(max_workers=1);drawn=draw();future=prefetch.submit(prepare,drawn[0]);last_update_end=started
    for step in range(1,2001):
        tick=time.monotonic();batch=dp.batch(future.result());orderhash.update(drawn[1]);used_rng_state=drawn[2]
        if step<2000:
            drawn=draw();future=prefetch.submit(prepare,drawn[0])
        if step==1:
            evidence=dp.evidence(batch);(O/f'critic_{args.method}_v2_sharding.json').write_text(json.dumps(evidence,indent=2));print('REAL_DATA_PARALLEL',json.dumps(evidence),flush=True)
        m={k:float(v) for k,v in jax.device_get(critic_step(critic,target,vopt,qopt,batch,config=cfg)).items()}
        end=time.monotonic();assert all(np.isfinite(v) for v in m.values()) and m['update_applied'];m.update(step=step,update_seconds=end-tick,end_to_end_step_seconds=end-last_update_end,samples_per_second=256/(end-last_update_end));metrics.append(m);last_update_end=end
        status=dict(method=args.method,step=step,total_steps=2000,elapsed_seconds=end-started,latest=m,sample_order_hash=orderhash.hexdigest(),execution=protocol['execution'],**initial)
        path=O/f'critic_{args.method}_v2_progress.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(status));tmp.replace(path)
        if step==1 or step%25==0:print('UPDATE',json.dumps(status),flush=True)
        if step in [250,500,1000,2000]:
            folder=(out/f'step{step:04d}').resolve()
            save_state=dict(critic=nnx.state(critic,nnx.Param).to_pure_dict(),target=nnx.state(target,nnx.Param).to_pure_dict(),value_optimizer=nnx.state(vopt).to_pure_dict(),q_optimizer=nnx.state(qopt).to_pure_dict())
            # Checkpoints store identical global FP32 values without binding
            # subsequent single-card evaluation/QAM to the training mesh.
            save_state=jax.tree.map(lambda x:np.asarray(jax.device_get(x)),save_state)
            with ocp.PyTreeCheckpointer() as ck:ck.save(folder,save_state)
            (out/f'metrics_step{step:04d}.json').write_text(json.dumps(dict(protocol=protocol,step=step,metrics=metrics,sample_order_hash=orderhash.hexdigest()),indent=2))
            (out/f'replay_rng_step{step:04d}.json').write_text(json.dumps(used_rng_state))
    prefetch.shutdown()
    (O/f'critic_{args.method}_v2_complete.json').write_text(json.dumps(dict(passed=True,steps=2000,protocol=protocol,sample_order_hash=orderhash.hexdigest(),elapsed_seconds=time.monotonic()-started,final=metrics[-1]),indent=2))
    print('CRITIC_V2_COMPLETE',args.method,flush=True)

if __name__=='__main__':main()
