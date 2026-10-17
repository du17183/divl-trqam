"""Two fixed replay mixes, unchanged critic objective/backbone/budget."""
from pathlib import Path
import argparse,copy,dataclasses,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np,flax.nnx as nnx,jax,orbax.checkpoint as ocp
from action_sensitive_v4_common import R,D,O,sha
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config,initialize_training,critic_step
from openpi.training.precision import assert_fp32_master
from critic_data_parallel import CriticDataParallel
from train_libero_lwd import digest,CRITIC_REPLAY_FIELDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);p.add_argument('--mix',choices=['R1','R2'],required=True)
    p.add_argument('--device-count',type=int,default=1);p.add_argument('--loader-workers',type=int,default=8);args=p.parse_args()
    assert jax.device_count()==args.device_count and 256%args.device_count==0
    assert json.loads((R/'artifacts/v4_positive_control.json').read_text())['passed']
    data=json.loads((D/'manifest.json').read_text());original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())
    assert not data.get('dev_states_used',False) and data['outcomes']==7*data['states']
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text());assert data['split_hash']==split['sha256']
    rows=original['records']+data['records'];episodes=[];identities=[];orig_indices=[];anchors=[]
    def load(row):
        assert sha(row['path'])==row['sha256']
        with np.load(row['path'],allow_pickle=False) as f:e={k:f[k] for k in CRITIC_REPLAY_FIELDS}
        assert int(e['init_id']) in split['tasks'][str(e['task_name'])]['train']
        return e
    with ThreadPoolExecutor(max_workers=args.loader_workers) as loader:
        for n,(row,e) in enumerate(zip(rows,loader.map(load,rows),strict=True)):
            episodes.append(e);identities.append(row['episode_id'])
            if n<len(original['records']):orig_indices.extend((n,i) for i in range(len(e['rewards'])))
            else:anchors.append((n,0))
            if n%250==0:print('V4_REPLAY_LOAD',n,len(rows),flush=True)
    orig_indices=np.asarray(orig_indices);anchors=np.asarray(anchors)
    cfg=LiberoTrainingConfig(value_type=args.method,reference_type='fixed',actor_updates=False)
    enc=encoder_config(R);tokenizer=DIVLTokenizer(enc);norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    critic=build_critic(cfg,enc);target,vopt,qopt=initialize_training(critic,2000,learning_rate=.0002)
    assert_fp32_master(critic);assert_fp32_master(target)
    initial=dict(q_initial_hash=digest(critic.q_head),encoder_initial_hash=digest(critic.encoder))
    dp=CriticDataParallel(critic,target,vopt,qopt,jax.devices());rng=np.random.default_rng(cfg.seed)
    branch_count={'R1':64,'R2':128}[args.mix];orderhash=hashlib.sha256();metrics=[];started=time.monotonic()
    out=R/'checkpoints/critic_replay_v4'/args.mix/args.method;assert not out.exists();out.mkdir(parents=True)
    protocol=dict(config=dataclasses.asdict(cfg),mix=args.mix,batch_size=256,td_chunks=10,steps=2000,seed=cfg.seed,learning_rate=.0002,
        branch_samples=branch_count,deployment_samples=256-branch_count,
        branch_sampling='Uniform among all retained v4 branch anchors, including flatreturn and failures; continuation is used for unchangedTD10 targets, not oversampled',
        execution=dict(data_parallel_devices=args.device_count,global_batch=256,microbatch_per_device=256//args.device_count,loader_workers=args.loader_workers,
                       gradient_reduction='GSPMD global mean / all-reduce',prefetch_batches=1,checkpoint_array_storage='Portable unchangedFP32 hostNumPy'),
        replay_manifest_sha256=sha(D/'manifest.json'),original_manifest_sha256=sha(R/'data/pilot_libero_v1/manifest.json'),
        actor_frozen=True,beta_dynamic=False,**initial)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def draw():
        indices=np.concatenate([orig_indices[rng.integers(len(orig_indices),size=256-branch_count)],anchors[rng.integers(len(anchors),size=branch_count)]])
        indices=indices[rng.permutation(256)];samples=[(episodes[e],int(i)) for e,i in indices]
        return samples,json.dumps([(identities[e],int(i)) for e,i in indices],separators=(',',':')).encode(),copy.deepcopy(rng.bit_generator.state)
    def prepare(samples):
        batch=make_batch(samples,tokenizer,norm,td_steps=10,return_numpy=True);batch['critic_k']=np.asarray([int(e['lengths'][i]) for e,i in samples],np.int32);return batch
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        drawn=draw();future=prefetch.submit(prepare,drawn[0]);last_end=started
        for step in range(1,2001):
            tick=time.monotonic();batch=dp.batch(future.result());orderhash.update(drawn[1]);used_rng=drawn[2]
            if step<2000:drawn=draw();future=prefetch.submit(prepare,drawn[0])
            if step==1:(O/f'critic_{args.mix}_{args.method}_v4_sharding.json').write_text(json.dumps(dp.evidence(batch),indent=2))
            m={k:float(v) for k,v in jax.device_get(critic_step(critic,target,vopt,qopt,batch,config=cfg)).items()};end=time.monotonic()
            assert all(np.isfinite(v) for v in m.values()) and m['update_applied']
            m.update(step=step,update_seconds=end-tick,end_to_end_step_seconds=end-last_end,samples_per_second=256/(end-last_end));last_end=end;metrics.append(m)
            status=dict(method=args.method,mix=args.mix,step=step,total_steps=2000,elapsed_seconds=end-started,latest=m,sample_order_hash=orderhash.hexdigest(),execution=protocol['execution'],**initial)
            dest=O/f'critic_{args.mix}_{args.method}_v4_progress.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(status));tmp.replace(dest)
            if step==1 or step%25==0:print('V4_UPDATE',json.dumps(status),flush=True)
            if step in [250,500,1000,2000]:
                save=dict(critic=nnx.state(critic,nnx.Param).to_pure_dict(),target=nnx.state(target,nnx.Param).to_pure_dict(),value_optimizer=nnx.state(vopt).to_pure_dict(),q_optimizer=nnx.state(qopt).to_pure_dict())
                save=jax.tree.map(lambda x:np.asarray(jax.device_get(x)),save)
                with ocp.PyTreeCheckpointer() as ck:ck.save((out/f'step{step:04d}').resolve(),save)
                (out/f'metrics_step{step:04d}.json').write_text(json.dumps(dict(protocol=protocol,step=step,metrics=metrics,sample_order_hash=orderhash.hexdigest()),indent=2))
                (out/f'replay_rng_step{step:04d}.json').write_text(json.dumps(used_rng))
    (O/f'critic_{args.mix}_{args.method}_v4_complete.json').write_text(json.dumps(dict(passed=True,steps=2000,protocol=protocol,sample_order_hash=orderhash.hexdigest(),elapsed_seconds=time.monotonic()-started,final=metrics[-1]),indent=2))
    print('CRITIC_V4_COMPLETE',args.mix,args.method,flush=True)


if __name__=='__main__':main()
