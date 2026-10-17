"""Independent SFT1000 actors: 100 updates only, full FP32, complete state I/O.

Native issued H5 is the critic input, actually executed cross-chunk H10 is BC.
No oracle/contact state enters the policy. No Dev result selects a checkpoint.
"""
from pilot8_common_v1 import *
import argparse
import gc
import traceback


def verify():
    for rel,h in read(EXP/'provenance/actor_resume_sources008.json')['files'].items():assert sha(ROOT/rel)==h,rel
    proof=read(EXP/'provenance/priority_cd_sources006.json')
    for rel,h in proof['files'].items():assert sha(ROOT/rel)==h,rel
    from critic_restore_current_layout_v1 import verify_repair
    verify_repair()
    from pilot8_critic_v1 import verify as upstream
    upstream()
    proof=read(EXP/'provenance/execution_sources001.json')
    for rel,h in proof['files'].items():assert sha(ROOT/rel)==h,rel
    assert read(EXP/'metrics/trqam_cpu_gate001.json')['passed']
    for rel,h in read(EXP/'provenance/critic_4plus4_sources003.json')['files'].items():assert sha(ROOT/rel)==h,rel


def checked_bc_actions(row):
    """H10 actual real7 labels already occupy padded32; no second padding."""
    import numpy as np
    actions = np.asarray(row['bc_actual_h10'], np.float32)
    mask = np.asarray(row['bc_actual_h10_mask'], np.float32)
    assert row['bc_full_h10_eligible'], 'Incomplete executed H10 window'
    assert actions.shape == (10, 32) and mask.shape == (10, 32), 'Expected padded32 actual H10'
    assert np.isfinite(actions).all() and np.isfinite(mask).all(), 'Nonfinite BC labels'
    assert np.all(mask[:, :7] == 1), 'Unexecuted action present in BC labels'
    assert np.all(mask[:, 7:] == 0), 'Padding must not be marked as executed'
    assert np.all(actions[:, 7:] == 0), 'Nonzero native FM padding'
    return actions.copy()


class BCData:
    def __init__(self,replay):
        import numpy as np
        c=config();self.norm=read(c['normalization']);m=read(c['train_manifest'])
        assert m['admitted'] and len(m['episodes'])==50 and sha(c['train_manifest'])==c['train_manifest_sha256']
        self.expert={t:[] for t in range(10)};self.rollout={t:[] for t in range(10)};self.replay=replay
        for e in m['episodes']:
            assert e['native_success'] and e['expert_training_eligible'];self.expert[e['task_id']].append(e)
        assert all(len(es)==5 for es in self.expert.values())
        for t,seq in replay.by_task.items():
            for r in seq:
                if not r['success']:continue
                # Select only full actual H10 labels; never predicted tails or mask-padding.
                replay.get(r['episode_id'],0);g=replay.handles[r['trajectory_path']][r['episode_id']]
                eligible=np.flatnonzero(g['chunks/bc_full_h10_eligible'][:])
                if len(eligible):self.rollout[t].append((r,eligible))

    def samples(self,number,batch):
        import numpy as np
        from benchmark_data_v1 import load_sample
        from benchmark_io_v1 import observation_data
        rng=np.random.default_rng(np.random.SeedSequence([config()['seed'],number,8105]));tasks=(np.arange(batch)+(number-1)*batch)%10;rng.shuffle(tasks)
        rows=[];indices=[]
        for t in tasks:
            t=int(t)
            if self.rollout[t] and rng.random()>=.5:
                r,eligible=self.rollout[t][int(rng.integers(len(self.rollout[t])))];i=int(rng.choice(eligible));row=self.replay.get(r['episode_id'],i)
                actions = checked_bc_actions(row)
                rows.append({'raw':row['raw'][0],'task_id':t,'actions':actions})
                indices.append({'source':'successful_rollout','task_id':t,'episode_id':r['episode_id'],'chunk_index':i,'state_source_id':r['state_source_id']})
            else:
                e=self.expert[t][int(rng.integers(5))];anchor=int(rng.integers(e['length']-9));row=load_sample((e,anchor,self.norm));rows.append(row)
                indices.append({'source':'legal_train_expert','task_id':t,'episode_id':e['unique_episode_id'],'anchor':anchor,'state_source_id':e['state_source_id']})
        return rows,indices


def stack(items):
    import numpy as np
    if isinstance(items[0],dict):return {k:stack([x[k] for x in items]) for k in items[0]}
    return np.stack(items)


def actor_observations(raw_rows,languages,tokenizer,norm):
    from benchmark_io_v1 import observation_data
    return stack([observation_data(r['raw'][0],r['raw'][0]['proprio'],languages[r['task_id']],tokenizer,norm) for r in raw_rows])


def bc_observations(rows,languages,tokenizer,norm):
    import numpy as np
    from benchmark_io_v1 import observation_data
    obs=[]
    for r in rows:
        if 'raw' in r:ob=observation_data(r['raw'],r['raw']['proprio'],languages[r['task_id']],tokenizer,norm)
        else:
            ids,mask=tokenizer.tokenize(languages[r['task_id']],state=None)
            ob={'images':r['images'],'state':r['state'],'image_masks':{k:np.asarray(k!='right_wrist_0_rgb') for k in r['images']},
                'tokenized_prompt':np.asarray(ids,np.int32),'tokenized_prompt_mask':np.asarray(mask,bool)}
        obs.append(ob)
    actions=np.stack([r['actions'] for r in rows]);assert actions.shape[1:]==(10,32) and np.all(actions[:,:,7:]==0)
    return stack(obs),actions


def make_bundle(method):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    from openpi.models.model import restore_params
    from openpi.models.pi0_config import Pi0Config
    c=config();a=read(EXP/'configs/actor_protocol001.json')
    with jax.default_device(jax.devices()[0]):
        params=restore_params(c['sft_checkpoint']+'/params',dtype=jnp.float32,sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
        actor=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,dtype='float32').load(params);actor.eval();reference=nnx.clone(actor)
        def optimizer(m):return nnx.Optimizer(m,optax.adamw(optax.cosine_decay_schedule(a['learning_rate'],100),b1=.9,b2=.999,weight_decay=.01),wrt=nnx.Param)
        opt=optimizer(actor);b=nnx.Dict(actor=actor,reference=reference,opt=opt)
        if method in ['G','H']:
            beta=nnx.clone(actor);b.beta=beta;b.betaopt=optimizer(beta);b.dual=nnx.Dict(lam=nnx.Variable(jnp.asarray(1.,jnp.float32)),kl_ema=nnx.Variable(jnp.asarray(0.,jnp.float32)))
    assert all(str(x.dtype)=='float32' for x in jax.tree.leaves(nnx.state(b.actor,nnx.Param)))
    return b


def place(b,devices):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from jax.sharding import Mesh,NamedSharding,PartitionSpec,SingleDeviceSharding
    assert devices in [1,4,8] and len(jax.devices())>=devices
    if devices==1:sh=data=SingleDeviceSharding(jax.devices()[0])
    else:
        mesh=Mesh(np.asarray(jax.devices()[:devices]),('pilot_batch',))
        sh=NamedSharding(mesh,PartitionSpec());data=NamedSharding(mesh,PartitionSpec('pilot_batch'))
    nnx.update(b,jax.tree.map(lambda x:jax.device_put(x,sh),nnx.state(b)))
    jax.block_until_ready(nnx.state(b));return data


def save(b,number,path,meta):
    import flax.nnx as nnx
    import numpy as np
    import orbax.checkpoint as ocp
    path=Path(path);assert path.resolve().is_relative_to((EXP/'checkpoints/actor').resolve());path.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with ocp.PyTreeCheckpointer() as w:
        w.save(str((path/'full_train_state').resolve()),{'state':nnx.state(b).to_pure_dict(),'update':np.asarray(number,np.int32)})
        w.save(str((path/'params').resolve()),nnx.state(b.actor,nnx.Param).to_pure_dict())
        if meta['method']=='B':w.save(str((path/'ema_params').resolve()),nnx.state(b.reference,nnx.Param).to_pure_dict())
    files={p.relative_to(path).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in path.rglob('*') if p.is_file()}
    m={**meta,'complete':True,'update':number,'next_update':number+1,'rng_rule':'fold_in(key(seed), update) with fixed per-role salts; reconstruct next update exactly',
        'files':files,'bytes':sum(x['bytes'] for x in files.values()),'wall_s':time.monotonic()-started,'saved_utc':stamp(),
        'full_components':['actor','reference','optimizer']+(['beta','beta_optimizer','lambda','kl_ema'] if meta['method'] in ['G','H'] else [])}
    write(path/'manifest.json',m,True);return m


def restore(b,path):
    import flax.nnx as nnx
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    m=read(Path(path)/'manifest.json');assert m['complete']
    for name,v in m['files'].items():assert sha(Path(path)/name)==v['sha256']
    state=nnx.state(b)
    template={'state':state.to_pure_dict(),'update':np.asarray(m['update'],np.int32)}
    from critic_restore_current_layout_v1 import restore_tree
    loaded=restore_tree(Path(path)/'full_train_state',template)
    assert int(loaded['update'])==m['update']
    state.replace_by_pure_dict(jax.tree.map(lambda x,y:jax.device_put(x,y.sharding),loaded['state'],state.to_pure_dict()));nnx.update(b,state)
    return m


def load_critic(method,devices):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from critic_model_v1 import build_pretrained
    from pilot8_critic_v1 import cfg
    value='scalar' if method in ['C','E','G'] else 'divl'
    if ENGINEERING_CRITIC:
        critic=build_pretrained(cfg(value));critic.eval();place(nnx.Dict(critic=critic),devices)
        from critic_fullbatch_v1 import parameter_digest
        return critic,'ENGINEERING_PUBLIC_INITIALIZATION:'+parameter_digest(nnx.state(critic))
    cp=EXP/'checkpoints/critic'/value/'step_002000';m=read(cp/'manifest.json');assert m['complete'] and m['update']==2000
    for name,v in m['files'].items():assert sha(cp/name)==v['sha256']
    with jax.default_device(jax.devices()[0]):
        critic=build_pretrained(cfg(value));state=nnx.state(critic,nnx.Param)
        from critic_restore_current_layout_v1 import restore_tree
        loaded=restore_tree(cp/'critic_params',state.to_pure_dict())
        state.replace_by_pure_dict(loaded);nnx.update(critic,state);critic.eval()
        holder=nnx.Dict(critic=critic);place(holder,devices)
    return critic,sha(cp/'manifest.json')


def get_batch(loader,bcdata,tok,ctok,number,batch):
    from critic_replay_v1 import collate
    rows,ridx=loader.samples(number,batch,config()['seed']);bcrows,bidx=bcdata.samples(number,batch)
    norm=read(config()['normalization']);obs=actor_observations(rows,loader.languages,tok,norm);bcobs,actions=bc_observations(bcrows,loader.languages,tok,norm)
    critic=collate(rows,loader.languages,ctok,norm)
    return {'obs':obs,'cobs':critic['obs'],'bcobs':bcobs,'bcactions':actions}, {'replay_index_sha256':ridx,'bc_indices':bidx,
        'bc_index_sha256':hashlib.sha256(json.dumps(bidx,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def run_update(b,critic,host,number,method,data):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    from openpi.training.qam_training import QAMTrainingConfig,QAMActionStats
    from pilot_actor_math_v1 import bc_step,qam_step,tr_step
    raw=jax.device_put(jax.tree.map(jnp.asarray,host),data)
    obs=Observation(**raw['obs']);bcobs=Observation(**raw['bcobs']);key=jax.random.fold_in(jax.random.key(config()['seed']),number)
    rk,bk=jax.random.split(key);stats=read(config()['normalization'])['statistics']['actions'];low=jnp.asarray(stats['q01']);high=jnp.asarray(stats['q99']);actionstats=QAMActionStats(low,high,low,high)
    if method=='B':out=bc_step(b.actor,b.reference,b.opt,bcobs,raw['bcactions'],bk,jnp.asarray(.005,jnp.float32))
    elif method in ['G','H']:out=tr_step(b.actor,b.beta,b.reference,critic,b.opt,b.betaopt,b.dual,obs,raw['cobs'],actionstats,rk,bcobs,raw['bcactions'],bk)
    else:
        c=QAMTrainingConfig(num_steps=10,epsilon=.1,temperature=2.,replan_steps=5,critic_horizon=5,action_dim=7,action_clip=None,sample_from_reference=True)
        out=qam_step(b.actor,b.reference,critic,b.opt,obs,raw['cobs'],actionstats,rk,bcobs,raw['bcactions'],bk,config=c,explicit_bc=method in ['E','F'])
    out=jax.block_until_ready(out);assert bool(out['update_applied']),'Nonfinite actor transaction rejected; stop branch'
    return out


def drift(b,host,data):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.model import Observation,restore_params
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import module_jit
    obs=Observation(**jax.device_put(jax.tree.map(jnp.asarray,host['obs']),data));noise=jax.random.normal(jax.random.key(20261012),(obs.state.shape[0],10,32))
    sample=module_jit(b.actor.sample_actions);actor=np.asarray(jax.device_get(sample(jax.random.key(0),obs,noise=noise,num_steps=10)))
    sample_ref=module_jit(b.reference.sample_actions);ref=np.asarray(jax.device_get(sample_ref(jax.random.key(0),obs,noise=noise,num_steps=10)))
    # Original SFT is loaded independently; B's EMA cannot be used as Frozen baseline.
    with jax.default_device(jax.devices()[0]):
        params=restore_params(config()['sft_checkpoint']+'/params',dtype=jnp.float32,sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
        original=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,dtype='float32').load(params);original.eval()
        place(nnx.Dict(actor=original),len(data.device_set));fresh=module_jit(original.sample_actions)
        old=np.asarray(jax.device_get(fresh(jax.random.key(0),obs,noise=noise,num_steps=10)))
    def param_rms(model):
        sq=0.;n=0
        for x,y in zip(jax.tree.leaves(nnx.state(model,nnx.Param)),jax.tree.leaves(nnx.state(original,nnx.Param)),strict=True):
            d=np.asarray(jax.device_get(x))-np.asarray(jax.device_get(y));sq+=float(np.sum(d.astype(np.float64)**2));n+=d.size
        return float(np.sqrt(sq/n))
    return {'actor_action_drift_rms_real7':float(np.sqrt(np.mean((actor[:,:,:7]-old[:,:,:7])**2))),
        'reference_action_drift_rms_real7':float(np.sqrt(np.mean((ref[:,:,:7]-old[:,:,:7])**2))),
        'actor_parameter_drift_rms':param_rms(b.actor),'reference_parameter_drift_rms':param_rms(b.reference),
        'scope':'same heldout raw observations and paired noise; normalized real7 H10; independent freshly restored SFT comparison'}


def gate(b,critic,loader,bcdata,tok,ctok,args,result,data):
    """Cross-device diagnostics separate from strict SAME-layout recovery.

    Cross-reduction is not subjected to a same-layout absolute threshold.
    Native/candidate global gradient norm and clipped Adam-mu direction must
    pass the predeclared mathematical scaling audit. All restoration checks
    retain original absolute2e-5. No change to scientific update functions.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.model import Observation
    from openpi.shared.nnx_utils import module_jit
    from critic_fullbatch_v1 import parameter_digest
    from critic_gpu_gate_contract_v1 import compare_snapshot
    from pilot8_critic_v1 import snapshot,host_metrics
    from pilot8_actor_numeric_review_v1 import cross_compare
    evidence=EXP/'data/gates'/args.run_id;evidence.mkdir(parents=True,exist_ok=False)
    b=make_bundle(args.method)
    result['initial_actor_sha256']=parameter_digest(nnx.state(b.actor,nnx.Param))
    result['initial_reference_sha256']=parameter_digest(nnx.state(b.reference,nnx.Param))
    assert result['initial_actor_sha256']==result['initial_reference_sha256']
    host,idx=get_batch(loader,bcdata,tok,ctok,1,args.batch);host2,idx2=get_batch(loader,bcdata,tok,ctok,2,args.batch)
    initial=parameter_digest(nnx.state(b));initial_critic=parameter_digest(nnx.state(critic)) if critic is not None else None
    data=place(b,1)
    if critic:place(nnx.Dict(critic=critic),1)
    native_metrics=host_metrics(run_update(b,critic,host,1,args.method,data))
    snapshot(nnx.state(b).to_pure_dict(),evidence/'native_state1')
    # The clipped first Adam moment provides a complete all-parameter gradient
    # direction check. Preclip norm is compared separately to detect scale bugs.
    del b;gc.collect();jax.clear_caches();b=make_bundle(args.method);data=place(b,args.devices)
    if critic:place(nnx.Dict(critic=critic),args.devices)
    assert parameter_digest(nnx.state(b))==initial
    obs=Observation(**jax.device_put(jax.tree.map(jnp.asarray,host['obs']),data))
    noise=jax.device_put(jax.random.normal(jax.random.key(20261013),(args.batch,10,32)),data)
    fresh=module_jit(b.actor.sample_actions)
    before=np.asarray(jax.device_get(jax.block_until_ready(fresh(jax.random.key(0),obs,noise=noise,num_steps=10))))
    del fresh
    candidate_metrics=host_metrics(run_update(b,critic,host,1,args.method,data))
    result['cross_layout_full_state_diagnostic']=cross_compare(nnx.state(b).to_pure_dict(),evidence/'native_state1')
    norm_keys=[k for k in native_metrics if 'grad_norm' in k]
    assert norm_keys and all(np.isfinite(native_metrics[k]) and native_metrics[k]>0 for k in norm_keys)
    ratios={k:candidate_metrics[k]/native_metrics[k] for k in norm_keys}
    assert all(abs(v-1)<=.001 for v in ratios.values()),('Global gradient scaling mismatch',ratios)
    result['gradient_scale_audit']={'native_metrics':native_metrics,'candidate_metrics':candidate_metrics,'preclip_norm_ratios':ratios,'normal_cross_device_differences_retained':True,'same_native_global32_mean_loss':True,'no_extra_1_over_device_count':True,'accumulation':False}
    assert result['cross_layout_full_state_diagnostic']['postclip_first_moment_cosine']>=.999999
    assert abs(result['cross_layout_full_state_diagnostic']['postclip_first_moment_norm_ratio']-1)<=.001
    if args.method in ['C','D','E','F']:assert parameter_digest(nnx.state(b.reference,nnx.Param))==result['initial_reference_sha256'],'Fixed reference changed'
    fresh=module_jit(b.actor.sample_actions)
    after=jax.block_until_ready(fresh(jax.random.key(0),obs,noise=noise,num_steps=10))
    action_drift=float(np.sqrt(np.mean((np.asarray(jax.device_get(after))[:,:,:7]-before[:,:,:7])**2)))
    assert np.isfinite(action_drift) and action_drift>0,'Updated actor output did not change'
    result['single_update_action_drift_real7_rms']=action_drift
    del fresh,before
    snapshot(after,evidence/'same_layout_inference1');del after
    snapshot(nnx.state(b).to_pure_dict(),evidence/'same_layout_state1')
    cp=EXP/'checkpoints/actor/gates'/args.run_id/'step_000001'
    save(b,1,cp,{'method':args.method,'scientific_actor':False})
    expected_digest=parameter_digest(nnx.state(b));result['candidate_state1_sha256']=expected_digest
    update2_metrics=host_metrics(run_update(b,critic,host2,2,args.method,data))
    snapshot(nnx.state(b).to_pure_dict(),evidence/'same_layout_state2')
    del b,obs,noise;gc.collect();jax.clear_caches();b=make_bundle(args.method);data=place(b,args.devices);restore(b,cp)
    result['same_layout_full_restore']=compare_snapshot(nnx.state(b).to_pure_dict(),evidence/'same_layout_state1')
    assert parameter_digest(nnx.state(b))==expected_digest,'Full actor/Adam/reference/beta/dual state restore mismatch'
    fresh=module_jit(b.actor.sample_actions)
    obs=Observation(**jax.device_put(jax.tree.map(jnp.asarray,host['obs']),data))
    noise=jax.device_put(jax.random.normal(jax.random.key(20261013),(args.batch,10,32)),data)
    result['fresh_restored_inference_comparison']=compare_snapshot(jax.block_until_ready(fresh(jax.random.key(0),obs,noise=noise,num_steps=10)),evidence/'same_layout_inference1')
    del fresh,obs,noise
    restored_metrics=host_metrics(run_update(b,critic,host2,2,args.method,data))
    result['restored_next_update']=compare_snapshot(nnx.state(b).to_pure_dict(),evidence/'same_layout_state2')
    for k in update2_metrics:
        if isinstance(update2_metrics[k],(float,int)):
            assert abs(restored_metrics[k]-update2_metrics[k])<=2e-5,('same_layout_metric',k)
    if critic is not None:assert parameter_digest(nnx.state(critic))==initial_critic,'Frozen critic changed'
    result.update(passed=True,full_pi05_parameters=True,fp32_master=True,fp32_compute=True,full_state_restore_passed=True,
        gradient_layout_checked_via_all_parameter_optimizer_updates=True,inference_jit_restore_checked=True,scientific_actor_updates=0,
        mathematical_updates='native full32 FM / terminal action gradient / backward adjoint; not -Q actor loss',absolute_tolerance=2e-5,
        cross_layout_same_restore_threshold_applied=False,physical_action_quality=False,sample_indices=[idx,idx2],disposable_actor_optimizer_updates=4,
        disposable_beta_optimizer_updates=4 if args.method in ['G','H'] else 0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=list('BCDEFGH'),required=True);p.add_argument('--mode',choices=['profile','gate','train'],required=True)
    p.add_argument('--batch',type=int,choices=[32],required=True);p.add_argument('--devices',type=int,choices=[1,4,8],required=True);p.add_argument('--run-id',required=True);p.add_argument('--engineering-critic',action='store_true');args=p.parse_args()
    global ENGINEERING_CRITIC
    ENGINEERING_CRITIC=args.engineering_critic
    assert not ENGINEERING_CRITIC or args.mode in ['profile','gate'],'No scientific actor update with fixture Critic'
    verify();dest=EXP/'metrics/actor'/args.run_id;dest.mkdir(parents=True,exist_ok=False)
    result={'passed':False,'method':args.method,'mode':args.mode,'batch':args.batch,'devices':args.devices,'start_utc':stamp(),
        'scientific_actor_updates':0,'scientific_critic_updates':0,'sourceproof_sha256':sha(EXP/'provenance/critic_4plus4_sources003.json'),
        'priority_sourceproof_sha256':sha(EXP/'provenance/priority_cd_sources006.json'),'resume_sourceproof_sha256':sha(EXP/'provenance/actor_resume_sources008.json'),
        'initial_checkpoint_manifest_sha256':config()['sft_checkpoint_manifest_sha256'],'replay_manifest_sha256':sha(EXP/'data/replay/manifest.json'),
        'actor_max_updates':100,'no_automatic_250_500':True,'engineering_critic_only':ENGINEERING_CRITIC}
    if args.mode=='profile':result['disposable_actor_optimizer_updates']=0
    start=time.monotonic();loader=None
    try:
        import flax.nnx as nnx
        import jax
        import numpy as np
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.models.divl_tokenizer import DIVLTokenizer
        from critic_model_v1 import encoder_config
        from critic_fullbatch_v1 import parameter_digest
        from pilot8_critic_v1 import Replay,Telemetry,receipt,host_metrics
        result['runtime_revision'] = runtime_receipt()
        jax.config.update('jax_default_matmul_precision','highest');result['gpu_receipt_before']=receipt()
        loader=Replay();bcdata=BCData(loader);tok=PaligemmaTokenizer(200);ctok=DIVLTokenizer(encoder_config())
        b=None;data=None;critic=None
        if args.mode!='gate':b=make_bundle(args.method);data=place(b,args.devices)
        if args.method!='B':critic,result['critic_manifest_sha256']=load_critic(args.method,args.devices)
        if b is not None:
            result['initial_actor_sha256']=parameter_digest(nnx.state(b.actor,nnx.Param));result['initial_reference_sha256']=parameter_digest(nnx.state(b.reference,nnx.Param))
            assert result['initial_actor_sha256']==result['initial_reference_sha256']
        result['critic_state_frozen_sha256']=parameter_digest(nnx.state(critic)) if critic is not None else None
        with Telemetry() as telemetry:
            if args.mode=='gate':gate(b,critic,loader,bcdata,tok,ctok,args,result,data)
            else:
                if args.mode=='train':
                    execution=read(EXP/f'configs/actor_execution_{args.method}001.json');assert execution['passed'] and execution['batch']==args.batch and execution.get('devices_by_method',{}).get(args.method,execution.get('devices'))==args.devices
                    gateid=execution['gates'][args.method];assert read(EXP/'metrics/actor'/gateid/'result.json')['passed'];assert sha(EXP/'metrics/actor'/gateid/'result.json')==execution['gate_result_sha256']
                    cp=EXP/'checkpoints/actor'/args.method;save(b,0,cp/'step_000000',{'method':args.method,'scientific_actor':True,**result})
                times=[];reads=[];drawhash=hashlib.sha256();bchash=hashlib.sha256();n=3 if args.mode=='profile' else 100
                for step in range(1,n+1):
                    assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD'
                    ts=time.monotonic();host,idx=get_batch(loader,bcdata,tok,ctok,step,args.batch);read_s=time.monotonic()-ts
                    ts=time.monotonic();metrics=run_update(b,critic,host,step,args.method,data);elapsed=time.monotonic()-ts
                    times.append(elapsed);reads.append(read_s);drawhash.update(idx['replay_index_sha256'].encode());bchash.update(idx['bc_index_sha256'].encode())
                    entry={'update':step,'method':args.method,'batch':args.batch,'read_s':read_s,'compute_s':elapsed,'total_s':elapsed+read_s,'gradient_clipped':float(metrics['grad_norm'])>1.,'learning_rate':read(EXP/'configs/actor_protocol001.json')['learning_rate']*.5*(1+np.cos(np.pi*(step-1)/100)),**idx,**host_metrics(metrics)}
                    with (dest/'train_steps.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                    if args.mode=='train':result['scientific_actor_updates']=step
                    else:result['disposable_actor_optimizer_updates']=step
                    write(dest/'lifecycle.json',{**result,'last_update':step,'utc':stamp()});print('ACTOR_UPDATE',json.dumps(entry),flush=True)
                if critic is not None:assert parameter_digest(nnx.state(critic))==result['critic_state_frozen_sha256'],'Actor updated critic unexpectedly'
                result.update(passed=True,steady_compute_s=float(np.mean(times[1:])),steady_read_s=float(np.mean(reads[1:])),
                    steady_total_s=float(np.mean(np.array(times[1:])+np.array(reads[1:]))),steady_samples_s=float(args.batch/np.mean(np.array(times[1:])+np.array(reads[1:]))),
                    sample_sequence_sha256=drawhash.hexdigest(),bc_sequence_sha256=bchash.hexdigest())
                if args.mode=='train':
                    # Make every accepted100-step update durable BEFORE optional diagnostics.
                    # Drift does not participate in optimization or checkpoint selection.
                    saved=save(b,100,EXP/'checkpoints/actor'/args.method/'step_000100',{'method':args.method,'scientific_actor':True,**result})
                    result['durable_actor100_manifest_sha256']=sha(EXP/'checkpoints/actor'/args.method/'step_000100/manifest.json')
                    result['checkpoint_saved_before_drift']=True
                    hold=Replay('holdout',verify_files=False);heldrows,_=hold.samples(1,16,20261012)
                    ahost={'obs':actor_observations(heldrows,loader.languages,tok,read(config()['normalization']))};result['drift']=drift(b,ahost,data);hold.close()
        result['telemetry']=telemetry.rows;result['gpu_receipt_after']=receipt()
    except Exception:
        result['passed']=False;result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if loader:loader.close()
        result.update(wall_s=time.monotonic()-start,ended_utc=stamp());result['allocated_gpu_hours']=args.devices*result['wall_s']/3600
        write(dest/'result.json',result,True);print('ACTOR_RESULT',json.dumps({k:v for k,v in result.items() if k!='telemetry'}),flush=True)
    sys.exit(0 if result['passed'] else 1)


if __name__=='__main__':main()
