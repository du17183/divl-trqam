"""Independent pilot Replay, full FP32 critic transactions and honest diagnostics.

Reuse the frozen native Scalar/DIVL equations, not the old project's readiness,
normalization, sampler manifest or checkpoint namespace. No actor is imported.
"""
from pilot8_common_v1 import *
import argparse
import dataclasses
import gc
import threading
import traceback


def verify():
    return verify_worker()


class Replay:
    def __init__(self, partition='train', verify_files=True):
        import h5py
        m = read(EXP/'data/replay/manifest.json')
        assert m['audit_passed'] and m['complete'] and m['schema']=='pilot2gpu_independently_audited_replay001'
        assert sha(m['episodes_path']) == m['episodes_sha256']
        self.records = read(m['episodes_path'])
        self.languages = [x['language'] for x in read(ROOT/'configs/libero_10_split_manifest.json')['tasks']]
        self.names = [x['task_name'] for x in read(ROOT/'configs/libero_10_split_manifest.json')['tasks']]
        self.by_task = {t: [] for t in range(10)}
        split = read(EXP/'configs/replay_train_holdout001.json')
        self.holdout = {t: set(split['tasks'][t]['critic_holdout_sources']) for t in range(10)}
        paths = set()
        self.by_id = {}; self.handles = {}
        for r in self.records:
            t = r['task_id']; held = r['state_source_id'] in self.holdout[t]
            selected = partition=='all' or (partition=='holdout')==held
            if selected and r['transitions'] > 0:
                self.by_task[t].append(r); self.by_id[r['episode_id']] = r
            paths.add(r['trajectory_path'])
        assert all(self.by_task.values()), 'A task has no eligible transitions; do not omit it'
        if verify_files:
            for f in m['files']:
                assert sha(f['hdf5']) == f['sha256'], f['hdf5']
        self.manifest_sha256 = sha(EXP/'data/replay/manifest.json')

    def get(self, episode, index):
        import h5py
        from critic_replay_v1 import extract_transition
        r = self.by_id[episode]; p = r['trajectory_path']
        if p not in self.handles: self.handles[p] = h5py.File(p, 'r')
        row = extract_transition(self.handles[p][episode], index, r['task_id'], episode)
        row['episode_success'] = r['success']
        return row

    def samples(self, update, batch, seed):
        from critic_replay_v1 import draw_indices
        draw = draw_indices(self.by_task, update, batch, seed)
        return [self.get(ep, idx) for _, ep, idx in draw], hashlib.sha256(json.dumps(draw,separators=(',',':')).encode()).hexdigest()

    def close(self):
        for f in self.handles.values(): f.close()
        self.handles.clear()


def receipt():
    return gpu_receipt()


def cfg(method):
    from critic_model_v1 import CriticConfig
    p=config()
    return CriticConfig(value_type=method,seed=p['seed'],learning_rate=p['critic_learning_rate'],
        expectile=p['scalar_expectile'],tau_base=p['divl']['tau_base'],alpha=p['divl']['entropy_alpha'],ema_rate=p['target_new_weight'])


def bundle(method):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from critic_model_v1 import build_pretrained, initialize_training
    # All initial masters, targets and optimizer moments go to physical GPU0.
    # The GPU1 context is visible but not used for single-device model computation.
    with jax.default_device(jax.devices()[0]):
        c=cfg(method); model=build_pretrained(c)
        target,vopt,qopt=initialize_training(model,c.updates,learning_rate=c.learning_rate)
        b=nnx.Dict(critic=model,target=target,vopt=vopt,qopt=qopt)
        sh=jax.sharding.SingleDeviceSharding(jax.devices()[0])
        nnx.update(b,jax.tree.map(lambda x:jax.device_put(x,sh),nnx.state(b)))
    assert all(x.dtype==np.dtype('float32') for x in jax.tree.leaves(nnx.state(b.critic,nnx.Param)))
    return b


def layout(b, devices):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding
    if devices==1: sh=data=SingleDeviceSharding(jax.devices()[0])
    else:
        assert devices in (2,4) and len(jax.devices())>=devices
        mesh=Mesh(np.asarray(jax.devices()[:devices]),('pilot_batch',))
        sh=NamedSharding(mesh,PartitionSpec());data=NamedSharding(mesh,PartitionSpec('pilot_batch'))
    nnx.update(b,jax.tree.map(lambda x:jax.device_put(x,sh),nnx.state(b)))
    jax.block_until_ready(nnx.state(b)); return data


def update(b, c, data, host):
    import jax
    import jax.numpy as jnp
    from critic_model_v1 import critic_step
    a=jax.device_put(jax.tree.map(jnp.asarray,host),data)
    out=jax.block_until_ready(critic_step(b.critic,b.target,b.vopt,b.qopt,a,config=c))
    assert bool(out['update_applied']), 'Nonfinite native transaction rejected; no scientific update counted'
    return out,a


def save(b, number, path, metadata):
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    from critic_state_v1 import payload
    path=Path(path); assert path.absolute().is_relative_to(EXP)
    assert path.resolve().is_relative_to((EXP/'checkpoints').resolve()) or path.resolve().is_relative_to((EXP/'data/gates').resolve())
    path.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    with ocp.PyTreeCheckpointer() as w:
        w.save(str((path/'full_train_state').resolve()),payload(b,number))
        w.save(str((path/'critic_params').resolve()),nnx.state(b.critic,nnx.Param).to_pure_dict())
    files={p.relative_to(path).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in path.rglob('*') if p.is_file()}
    m={**metadata,'complete':True,'update':number,'next_sampler_update':number+1,'files':files,'saved_utc':stamp(),
       'save_and_hash_wall_s':time.monotonic()-start,'bytes':sum(x['bytes'] for x in files.values()),
       'full_state_components':['online_encoder_q_v','target_encoder_q_v','value_adam','q_adam','update']}
    write(path/'manifest.json',m,True);return m


def host_metrics(m):
    import jax
    import numpy as np
    return {k:float(np.asarray(jax.device_get(v))) for k,v in m.items()}


def snapshot(tree,path):
    """Same absolute full-leaf evidence contract, in this pilot's new namespace."""
    import jax
    import numpy as np
    path=Path(path);assert path.resolve().is_relative_to((EXP/'data/gates').resolve())
    path.mkdir(parents=True,exist_ok=False);rows=[]
    for number,(key,leaf) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
        a=np.asarray(jax.device_get(leaf));assert np.isfinite(a).all();p=path/f'{number:06d}.npy'
        with p.open('xb') as f:np.save(f,a,allow_pickle=False)
        rows.append({'key':jax.tree_util.keystr(key),'file':p.name,'shape':list(a.shape),'dtype':str(a.dtype),'bytes':p.stat().st_size,'sha256':sha(p)})
    write(path/'manifest.json',{'leaves':rows,'all_leaves_saved':True,'absolute_tolerance':2e-5,'bytes':sum(x['bytes'] for x in rows)},True)


class Telemetry:
    def __init__(self): self.rows=[]; self.stop=threading.Event(); self.t=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set():
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
            own={}
            for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
                pid,uid,mem=[x.strip() for x in row.split(',')]
                if int(pid)==os.getpid():own[uid]=float(mem)
            for line in text.splitlines():
                i,u,m,z,p=[s.strip() for s in line.split(',')]
                if int(i) in physical_gpus():
                    assert u==UUIDS[int(i)];self.rows.append({'utc':stamp(),'gpu':int(i),'memory_mib':own.get(u,0.),'whole_card_memory_mib_diagnostic_only':float(m),'util_percent':float(z),'power_w':None,'whole_card_power_w_diagnostic_only':float(p),'scope':'Own process memory; whole-card utilization diagnostic only, no exclusive power estimate'})
            self.stop.wait(2)
    def __enter__(self):self.t.start();return self
    def __exit__(self,*_):self.stop.set();self.t.join()


def gate(args, loader, tok, norm, result):
    """Full encoder, raw gradients, optimizer/target state and fresh restore/JIT.

    Two-GPU DP is optional; if it fails the absolute gate the selector can admit
    the native one-GPU transaction after its independent restore gate succeeds.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from critic_replay_v1 import collate
    from critic_fullbatch_v1 import parameter_digest
    from critic_pretrained_gpu_gate_v1 import bind_gradient_probe, bind_inference
    from critic_gpu_gate_contract_v1 import compare_snapshot,compare_array
    from critic_model_v1 import action_gradient
    from critic_state_v1 import restore
    evidence=EXP/'data/gates'/args.run_id;evidence.mkdir(parents=True,exist_ok=False)
    c=cfg(args.method);host={};draw={}
    for i in (1,2,3):
        rows,draw[i]=loader.samples(i,args.batch,c.seed);host[i]=collate(rows,loader.languages,tok,norm)
    result['sample_indices_sha256']=draw
    b=bundle(args.method);data=layout(b,1);before=parameter_digest(nnx.state(b))
    result['initial_encoder_sha256']=parameter_digest(nnx.state(b.critic.encoder,nnx.Param))
    result['initial_q_sha256']=parameter_digest(nnx.state(b.critic.q_head,nnx.Param))
    probe=bind_gradient_probe(c);a=jax.device_put(jax.tree.map(jnp.asarray,host[1]),data)
    grads=jax.block_until_ready(probe(b,a));assert parameter_digest(nnx.state(b))==before
    assert all(np.isfinite(np.asarray(jax.device_get(x))).all() for x in jax.tree.leaves(grads))
    snapshot(grads,evidence/'reference_gradients');del grads
    m={}
    for i in (1,2):m[i]=host_metrics(update(b,c,data,host[i])[0])
    snapshot(nnx.state(b).to_pure_dict(),evidence/'reference_state2')
    a=jax.device_put(jax.tree.map(jnp.asarray,host[3]),data);infer=bind_inference()
    snapshot(jax.block_until_ready(infer(b.critic,a['obs'],a['actions'],a['action_mask'])),evidence/'reference_inference2')
    snapshot(jax.block_until_ready(action_gradient(b.critic,a)),evidence/'reference_action_gradient2')
    meta={'value_type':args.method,'seed':c.seed,'global_batch':args.batch,'devices':args.devices,'scientific_critic':False,
          'protocol_sha256':sha(EXP/'configs/protocol001.json'),'replay_manifest_sha256':loader.manifest_sha256,'compute_dtype':'float32','master_dtype':'float32'}
    refsaved=None
    if args.devices==1:refsaved=save(b,2,evidence/'native_step2',meta)
    m[3]=host_metrics(update(b,c,data,host[3])[0]);snapshot(nnx.state(b).to_pure_dict(),evidence/'reference_state3')
    del b,a,infer,probe;gc.collect();jax.clear_caches()
    b=bundle(args.method);data=layout(b,args.devices)
    if args.devices>1:
        assert parameter_digest(nnx.state(b))==before
        probe=bind_gradient_probe(c);a=jax.device_put(jax.tree.map(jnp.asarray,host[1]),data)
        result['raw_gradient_comparison']=compare_snapshot(jax.block_until_ready(probe(b,a)),evidence/'reference_gradients')
        assert parameter_digest(nnx.state(b))==before
        for i in (1,2):
            measured,_=update(b,c,data,host[i]);result[f'metrics{i}']={k:compare_array(m[i][k],v) for k,v in host_metrics(measured).items()}
        result['full_state2_comparison']=compare_snapshot(nnx.state(b).to_pure_dict(),evidence/'reference_state2')
        path=evidence/'candidate_step2';save(b,2,path,meta)
        del b;gc.collect();jax.clear_caches();b=bundle(args.method);data=layout(b,args.devices)
    else: path=evidence/'native_step2'
    restored=restore(b,path,expected_metadata=meta);assert restored['next_sampler_update']==3
    # Inference binding is made only after FP32 restoration and final device placement.
    infer=bind_inference();a=jax.device_put(jax.tree.map(jnp.asarray,host[3]),data)
    result['restored_inference_comparison']=compare_snapshot(jax.block_until_ready(infer(b.critic,a['obs'],a['actions'],a['action_mask'])),evidence/'reference_inference2')
    result['restored_action_gradient_comparison']=compare_snapshot(jax.block_until_ready(action_gradient(b.critic,a)),evidence/'reference_action_gradient2')
    measured,_=update(b,c,data,host[3])
    result['restored_next_update_comparison']=compare_snapshot(nnx.state(b).to_pure_dict(),evidence/'reference_state3')
    measured_metrics = host_metrics(measured)
    result['restored_metrics_values'] = {k: {'reference': m[3][k], 'restored': v,
        'absolute_error': abs(m[3][k] - v)} for k, v in measured_metrics.items()}
    result['restored_metrics']={k:compare_array(m[3][k],v) for k,v in measured_metrics.items()}
    before=parameter_digest(nnx.state(b));bad={**a,'rewards':jnp.full_like(a['rewards'],jnp.nan)}
    from critic_model_v1 import critic_step
    rejected=jax.block_until_ready(critic_step(b.critic,b.target,b.vopt,b.qopt,bad,config=c))
    assert not bool(rejected['update_applied']) and parameter_digest(nnx.state(b))==before
    result.update(passed=True,full_pretrained_encoder=True,raw_gradients_checked=True,full_state_restore_jit_passed=True,
        layout_equivalence_passed=True,nonfinite_complete_rollback=True,absolute_tolerance=2e-5,
        physical_action_quality_proof=False,scientific_critic_updates=0,disposable_critic_transactions=4 if args.devices==1 else 6,
        temporary_value_probe_transactions=1 if args.devices==1 else 2)


def bind_readonly_diagnostics(c):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.training.libero_lwd_training import divl_q_target,expectile_loss,distributional_loss,adaptive_quantile_level
    from openpi.models.divl_losses import twin_q_loss
    from critic_model_v1 import action_gradient
    @nnx.jit
    def infer(model,target,batch):
        z=model.encoder(**batch['obs'],train=False);q1,q2=model.q_head(z,batch['actions'],batch['action_mask'])
        tz=target.encoder(**batch['obs'],train=False);qtarget=jnp.minimum(*target.q_head(tz,batch['actions'],batch['action_mask']))
        nv=target.v_head(target.encoder(**batch['next_obs'],train=False));values=model.v_head(z)
        if c.value_type=='scalar':
            td=batch['rewards']+batch['discounts']*nv
            vl=jax.vmap(lambda v,q:expectile_loss(v,q,c.expectile))(values,qtarget)
            entropy=jnp.zeros_like(q1);tau=jnp.full_like(q1,c.expectile)
        else:
            td=divl_q_target(batch['rewards'],batch['discounts'],nv,target.v_head.atoms,tau_base=c.tau_base,alpha=c.alpha,tau_min=0,tau_max=1)
            vl=jax.vmap(lambda v,q:distributional_loss(v,q,model.v_head.atoms))(values,qtarget)
            probs=jax.nn.softmax(nv.astype(jnp.float32),axis=-1)
            entropy=-jnp.sum(probs*jnp.log(jnp.maximum(probs,1e-10)),axis=-1)
            tau=adaptive_quantile_level(nv,tau_base=c.tau_base,alpha=c.alpha,tau_min=0,tau_max=1)
        grad=action_gradient(model,batch)
        return {'q1':q1,'q2':q2,'min_q':jnp.minimum(q1,q2),'td_error':jnp.abs(jnp.minimum(q1,q2)-td),
            'loss_q':jax.vmap(twin_q_loss)(q1,q2,td),'loss_v':vl,'action_gradient_norm':jnp.linalg.norm(grad.reshape((grad.shape[0],-1)),axis=-1),
            'value_entropy':entropy,'tau':tau}
    return infer

def diagnostics(b, loader, tok, norm, data, args, result):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import spearmanr,pearsonr
    from critic_replay_v1 import collate
    from critic_pretrained_gpu_gate_v1 import bind_inference
    from critic_fullbatch_v1 import parameter_digest
    import flax.nnx as nnx
    before=parameter_digest(nnx.state(b))
    infer=bind_readonly_diagnostics(cfg(args.method));records=[]
    for i in range(1,11):
        rows,_=loader.samples(i,100,20261011);host=collate(rows,loader.languages,tok,norm)
        batch=jax.device_put(jax.tree.map(jnp.asarray,host),data)
        detail=jax.device_get(jax.block_until_ready(infer(b.critic,b.target,batch)));q=np.asarray(detail['min_q'])
        assert all(np.isfinite(v).all() for v in detail.values()),'Nonfinite readonly diagnostic'
        for offset,(row,value) in enumerate(zip(rows,q,strict=True)):
            g=loader.handles[loader.by_id[row['episode_id']]['trajectory_path']][row['episode_id']]
            at=row['observation_indices'][0];rw=np.asarray(g['native_rewards'][at:],float)
            observed=float(np.dot(rw,.99**np.arange(len(rw))))
            records.append({'task_id':row['task_id'],'episode_id':row['episode_id'],'state_source_id':row['state_source_id'],
                'chunk_index':row['chunk_index'],'q':float(value),'observed_discounted_return':observed,
                'complete_return_known':not row['collection_censored'],'success':row['episode_success'],
                **{key:float(np.asarray(values)[offset]) for key,values in detail.items() if key not in ['q1','q2','min_q']}})
    per=[]
    for t in range(10):
        rr=[r for r in records if r['task_id']==t];q=np.array([r['q'] for r in rr]);g=np.array([r['observed_discounted_return'] for r in rr])
        ok=[r['q'] for r in rr if r['success']];bad=[r['q'] for r in rr if not r['success']]
        per.append({'task_id':t,'task_name':loader.names[t],'sampled_windows':len(rr),'independent_initial_sources':len({r['state_source_id'] for r in rr}),
            'independent_simulator_states':len({(r['episode_id'],r['chunk_index']) for r in rr}),
            'q_observed_return_mse':float(np.mean((q-g)**2)),'spearman_observed_suffix_return':float(spearmanr(q,g).statistic) if np.std(g)>0 and np.std(q)>0 else None,
            'pearson_observed_suffix_return':float(pearsonr(q,g).statistic) if np.std(g)>0 and np.std(q)>0 else None,
            'mean_q_success':float(np.mean(ok)) if ok else None,'mean_q_failure':float(np.mean(bad)) if bad else None,
            'pairwise_action_ranking':None,'near_id_action_ranking':None,'state_wise_spearman':None,'valid_action_pairs':0,
            **{key:float(np.mean([r[key] for r in rr])) for key in ['td_error','loss_q','loss_v','action_gradient_norm','value_entropy','tau']},
            'positive_gradient_vs_random_true_return':None,'reason':'No same-state physical counterfactual actions admitted; different states cannot form action-ranking pairs. Censored suffix return is observed, not complete MC.'})
    assert parameter_digest(nnx.state(b))==before,'Readonly diagnostics changed critic/optimizer/target'
    result.update(per_task=per,records=records,macro_q_observed_return_mse=float(np.mean([x['q_observed_return_mse'] for x in per])),
        action_ranking=None,true_physical_gradient=None,gate_for_extended_actor_training=False,
        actor_pilot_max_updates=100,actor_scope='Engineering pilot only; no physical action ranking/gradient admission; no250/500 automatic expansion')


def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['profile','gate','train','diagnose'],required=True)
    a.add_argument('--method',choices=['scalar','divl'],required=True);a.add_argument('--batch',type=int,choices=[64,128,256],required=True)
    a.add_argument('--devices',type=int,choices=[1,2,4],required=True);a.add_argument('--run-id',required=True);args=a.parse_args()
    verify();dest=EXP/'metrics/critic'/args.run_id;dest.mkdir(parents=True,exist_ok=False)
    result={'mode':args.mode,'method':args.method,'batch':args.batch,'devices':args.devices,'passed':False,'start_utc':stamp(),
            'scientific_critic_updates':0,'actor_updates':0,'replay_manifest_sha256':sha(EXP/'data/replay/manifest.json'),
            'sourceproof_sha256':sha(EXP/'provenance/execution_sources001.json'),'same_native_math':True,'gradient_accumulation':1,
            'layout':'replicated weights/data parallel' if args.devices>1 else 'native single logicalGPU0'}
    started=time.monotonic();loader=None
    try:
        import flax.nnx as nnx
        import jax
        import numpy as np
        from critic_model_v1 import encoder_config
        from critic_replay_v1 import collate
        from critic_fullbatch_v1 import parameter_digest
        from openpi.models.divl_tokenizer import DIVLTokenizer
        for f in read(ROOT/'provenance/critic_public_initialization001.json')['files']:assert sha(f['path'])==f['sha256']
        result['runtime_revision'] = runtime_receipt()
        jax.config.update('jax_default_matmul_precision','highest');result['gpu_receipt_before']=receipt()
        loader=Replay('holdout' if args.mode=='diagnose' else 'train');tok=DIVLTokenizer(encoder_config());norm=read(config()['normalization'])
        with Telemetry() as telemetry:
            if args.mode=='gate':gate(args,loader,tok,norm,result)
            else:
                if args.mode in ['train','diagnose']:
                    selected=read(EXP/'configs/critic_execution001.json');assert selected['passed'] and selected['batch']==args.batch and selected['devices']==args.devices and args.batch==128
                    assert all(read(EXP/'metrics/critic'/x/'result.json')['passed'] for x in selected['gates'].values())
                c=cfg(args.method);b=bundle(args.method);data=layout(b,args.devices)
                result['initial_encoder_sha256']=parameter_digest(nnx.state(b.critic.encoder,nnx.Param));result['initial_q_sha256']=parameter_digest(nnx.state(b.critic.q_head,nnx.Param))
                if args.mode=='diagnose':
                    from critic_state_v1 import restore
                    cp=EXP/'checkpoints/critic'/args.method/'step_002000';restore(b,cp,expected_metadata={'value_type':args.method,'global_batch':args.batch})
                    diagnostics(b,loader,tok,norm,data,args,result);result['passed']=True
                else:
                    steps=5 if args.mode=='profile' else 2000;times=[];readtimes=[];sampler=hashlib.sha256()
                    for number in range(1,steps+1):
                        assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD'
                        tm=time.monotonic();rows,indexsha=loader.samples(number,args.batch,c.seed);host=collate(rows,loader.languages,tok,norm);rd=time.monotonic()-tm
                        tm=time.monotonic();m,_=update(b,c,data,host);dt=time.monotonic()-tm;v=host_metrics(m);times.append(dt);readtimes.append(rd);sampler.update(indexsha.encode())
                        entry={'update':number,'batch':args.batch,'sample_index_sha256':indexsha,'read_s':rd,'compute_s':dt,'total_s':dt+rd,**v}
                        with (dest/'train_steps.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                        if args.mode=='train':result['scientific_critic_updates']=number
                        else:result['disposable_critic_transactions']=number
                        print('CRITIC_UPDATE',json.dumps(entry),flush=True);write(dest/'lifecycle.json',{**result,'last_update':number,'utc':stamp()})
                        if args.mode=='train' and number in [250,500,1000,2000]:
                            save(b,number,EXP/'checkpoints/critic'/args.method/f'step_{number:06d}',{
                                'value_type':args.method,'seed':c.seed,'global_batch':args.batch,'devices':args.devices,'compute_dtype':'float32','master_dtype':'float32',
                                'protocol_sha256':sha(EXP/'configs/protocol001.json'),'replay_manifest_sha256':loader.manifest_sha256,
                                'normalization_sha256':sha(config()['normalization']),'sample_sequence_sha256':sampler.hexdigest(),'scientific_critic':True})
                    result.update(passed=True,steps=steps,sample_sequence_sha256=sampler.hexdigest(),steady_compute_s=float(np.mean(times[1:])),
                        steady_read_s=float(np.mean(readtimes[1:])),steady_total_s=float(np.mean(np.array(times[1:])+np.array(readtimes[1:]))),
                        steady_samples_s=float(args.batch/np.mean(np.array(times[1:])+np.array(readtimes[1:]))))
        result['telemetry']=telemetry.rows;result['gpu_receipt_after']=receipt()
    except Exception:
        result['passed']=False;result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        if loader:loader.close()
        result.update(wall_s=time.monotonic()-started,ended_utc=stamp());result['allocated_gpu_hours']=args.devices*result['wall_s']/3600
        write(dest/'result.json',result,True);print('CRITIC_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['records','telemetry']}),flush=True)
    sys.exit(0 if result['passed'] else 1)


if __name__=='__main__':main()
