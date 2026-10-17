"""Candidate GPU gate; no auto-launch, no scientific critic or actor updates.

Actual B256 public Gemma3/SigLIP + selected-policy Replay are required. A single
GPU native full-batch reference is compared with a candidate replicated layout.
OOM or absolute error >2e-5 is a failed candidate, not an excuse to alter science.
Only an independent future selector may publish canonical admission files.
"""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def bind_gradient_probe(config):
    """Gate-only raw gradients from the exact native sequential V/Q equations.

Temporary V optimizer state is restored before return. This is computational
layout validation, not proof that a physical action improves environment return.
"""
    from functools import partial
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.training.libero_lwd_training import expectile_loss
    from openpi.training.divl_lwd_training import _prepare_gradients, _guarded_update
    from openpi.models.divl_losses import distributional_loss, divl_q_target, twin_q_loss

    @partial(nnx.jit, static_argnames=('config',))
    def raw(critic, target, value_opt, q_opt, batch, *, config):
        critic.train(); target.eval()
        bundle = nnx.Dict(critic=critic, value_optimizer=value_opt)
        before = nnx.state(bundle)
        z = target.encoder(**batch['obs'], train=False)
        qtarget = jax.lax.stop_gradient(jnp.minimum(*target.q_head(z, batch['actions'], batch['action_mask'])))
        def vloss(model):
            values = model.v_head(model.encoder(**batch['obs'], train=True))
            if config.value_type == 'scalar':
                return expectile_loss(values, qtarget, config.expectile)
            return distributional_loss(values, qtarget, model.v_head.atoms)
        lv, gv = nnx.value_and_grad(vloss, argnums=nnx.DiffState(0, value_opt.wrt))(critic)
        prepared, _, valid = _prepare_gradients(gv, lv, config.max_grad_norm)
        _guarded_update(value_opt, prepared, valid)
        next_value = target.v_head(target.encoder(**batch['next_obs'], train=False))
        td = (jax.lax.stop_gradient(batch['rewards']+batch['discounts']*next_value)
              if config.value_type == 'scalar' else
              divl_q_target(batch['rewards'], batch['discounts'], next_value, target.v_head.atoms,
                            tau_base=config.tau_base, alpha=config.alpha, tau_min=0, tau_max=1))
        def qloss(model):
            z = model.encoder(**batch['obs'], train=True)
            q1, q2 = model.q_head(z, batch['actions'], batch['action_mask'])
            return twin_q_loss(q1, q2, td)
        _, gq = nnx.value_and_grad(qloss, argnums=nnx.DiffState(0, q_opt.wrt))(critic)
        nnx.update(bundle, before)
        return {'value':gv, 'q':gq}
    return lambda bundle, batch: raw(bundle.critic, bundle.target, bundle.vopt, bundle.qopt, batch, config=config)


def bind_inference():
    import flax.nnx as nnx
    import jax.numpy as jnp
    @nnx.jit
    def infer(model, obs, actions, mask):
        z = model.encoder(**obs, train=False)
        q1, q2 = model.q_head(z, actions, mask)
        return {'q1':q1, 'q2':q2, 'min_q':jnp.minimum(q1,q2), 'value':model.v_head(z)}
    return infer  # Always pass the restored model explicitly; no captured parameter arrays.


def run(args):
    from critic_state_v1 import readiness
    if args.mode == 'status':
        print(json.dumps({'upstream':readiness(), 'gpu_initialized':False,
            'workers_launched':False, 'scientific_critic_updates':0,
            'full_pretrained_gpu_gate_executed':False, 'candidate_only':True}))
        return
    from critic_gpu_gate_contract_v1 import preflight, snapshot, compare_snapshot, compare_array, sha
    from critic_state_v1 import atomic, save, restore
    assert args.run_id and args.run_id.replace('_','').replace('-','').isalnum()
    gpus = [int(v) for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(gpus) == args.devices
    prereq, identities = preflight(gpus)  # No JAX/model initialization before this gate.
    destination = ROOT/'metrics/critic_pretrained_gpu_candidates'/args.run_id
    evidence = ROOT/'data/gates/critic_pretrained_gpu_candidates'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema':'critic_full_pretrained_gpu_candidate001', 'passed':False,
        'run_id':args.run_id, 'method':args.method, 'devices':gpus, 'pid':os.getpid(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prerequisites':prereq, 'identity_and_resource_preflight':identities,
        'global_batch':256, 'native_full_batch_only':True, 'gradient_accumulation_steps':1,
        'full_pretrained_encoder':True, 'real_policy_replay':True,
        'absolute_tolerance':2e-5, 'relative_error_used':False,
        'scientific_critic_updates':0, 'actor_updates':0,
        'physics_ranking_admitted':False, 'true_physical_gradient_validated':False,
        'canonical_admission_published':False, 'candidate_disposable_updates':0,
        'scope':'computational equivalence/restore/JIT only; no learned action-quality claim'}
    loader = None
    try:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import SingleDeviceSharding
        from critic_model_v1 import CriticConfig, encoder_config, critic_step, action_gradient
        from critic_fullbatch_v1 import make_bundle, bind_update, replicated_layout, parameter_digest
        from critic_replay_v1 import AuditedReplay, collate
        from openpi.models.divl_tokenizer import DIVLTokenizer
        assert len(jax.devices()) == args.devices and all(d.platform == 'gpu' for d in jax.devices())
        jax.config.update('jax_default_matmul_precision','highest')
        jax.config.update('jax_compilation_cache_dir', str(ROOT/'cache/critic_pretrained_gpu_gate'))
        p = prereq['protocol']
        cfg = CriticConfig(value_type=args.method, seed=p['seed_pilot'], expectile=p['scalar_expectile'],
            tau_base=p['divl']['tau_base'], alpha=p['divl']['entropy_alpha'],
            ema_rate=p['target_new_weight'], learning_rate=p['learning_rate'], max_grad_norm=p['gradient_clip'])
        loader = AuditedReplay(prereq['replay_manifest_path'])
        tokenizer = DIVLTokenizer(encoder_config())
        norm = json.loads((ROOT/'data/sft/common_norm_stats_v1.json').read_text())
        host, index_sha = {}, {}
        for number in [1,2,3]:
            rows, index_sha[number] = loader.samples(number,256,cfg.seed)
            host[number] = collate(rows,loader.languages,tokenizer,norm)
        result['sample_indices_sha256'] = index_sha
        single = SingleDeviceSharding(jax.devices()[0])
        metadata = {'value_type':args.method, 'seed':cfg.seed, 'scientific_critic':False,
            'fixture':False, 'global_batch':256, 'compute_dtype':'float32', 'master_dtype':'float32',
            'protocol_sha256':prereq['protocol_sha256'], 'replay_manifest_sha256':prereq['replay_manifest_sha256'],
            'execution_sources_sha256':prereq['execution_sources_sha256']}
        reference_metrics, reference_inference = {}, None
        probe = bind_gradient_probe(cfg)
        with jax.default_device(jax.devices()[0]):
            bundle = make_bundle(cfg)
            nnx.update(bundle, jax.tree.map(lambda x:jax.device_put(x,single),nnx.state(bundle)))
            result['initial_q_sha256'] = parameter_digest(nnx.state(bundle.critic.q_head,nnx.Param))
            result['initial_encoder_sha256'] = parameter_digest(nnx.state(bundle.critic.encoder,nnx.Param))
            initial_full = parameter_digest(nnx.state(bundle))
            batch = jax.device_put(jax.tree.map(jnp.asarray,host[1]),single)
            raw = jax.block_until_ready(probe(bundle,batch))
            assert parameter_digest(nnx.state(bundle)) == initial_full, 'Gradient probe changed training state'
            snapshot(raw,evidence/'reference_raw_gradients'); del raw
            for number in [1,2]:
                batch = jax.device_put(jax.tree.map(jnp.asarray,host[number]),single)
                metrics = jax.block_until_ready(critic_step(bundle.critic,bundle.target,bundle.vopt,bundle.qopt,batch,config=cfg))
                assert bool(metrics['update_applied'])
                reference_metrics[number] = {k:np.asarray(jax.device_get(v)) for k,v in metrics.items()}
            assert parameter_digest(nnx.state(bundle)) != initial_full
            snapshot(nnx.state(bundle).to_pure_dict(),evidence/'reference_state2')
            batch = jax.device_put(jax.tree.map(jnp.asarray,host[3]),single)
            infer = bind_inference()
            reference_inference = jax.block_until_ready(infer(bundle.critic,batch['obs'],batch['actions'],batch['action_mask']))
            reference_inference = jax.tree.map(lambda x:np.asarray(jax.device_get(x)),reference_inference)
            snapshot(reference_inference,evidence/'reference_inference2')
            reference_action_gradient = jax.block_until_ready(action_gradient(bundle.critic,batch))
            snapshot(reference_action_gradient,evidence/'reference_action_gradient2'); del reference_action_gradient
            metrics = jax.block_until_ready(critic_step(bundle.critic,bundle.target,bundle.vopt,bundle.qopt,batch,config=cfg))
            reference_metrics[3] = {k:np.asarray(jax.device_get(v)) for k,v in metrics.items()}
            snapshot(nnx.state(bundle).to_pure_dict(),evidence/'reference_state3')
        del bundle, batch, metrics, infer, probe
        gc.collect(); jax.clear_caches()
        bundle = make_bundle(cfg)
        layout = replicated_layout(bundle)
        assert parameter_digest(nnx.state(bundle.critic.q_head,nnx.Param)) == result['initial_q_sha256']
        assert parameter_digest(nnx.state(bundle.critic.encoder,nnx.Param)) == result['initial_encoder_sha256']
        assert parameter_digest(nnx.state(bundle)) == initial_full
        fn = bind_update(bundle,cfg)
        probe = bind_gradient_probe(cfg)
        batch = jax.device_put(jax.tree.map(jnp.asarray,host[1]),layout)
        raw = jax.block_until_ready(probe(bundle,batch))
        result['raw_gradient_comparison'] = compare_snapshot(raw,evidence/'reference_raw_gradients')
        assert parameter_digest(nnx.state(bundle)) == initial_full
        del raw
        result['metrics_comparisons'] = {}
        for number in [1,2]:
            batch = jax.device_put(jax.tree.map(jnp.asarray,host[number]),layout)
            metrics = jax.block_until_ready(fn(batch))
            assert bool(metrics['update_applied'])
            result['metrics_comparisons'][str(number)] = {k:compare_array(reference_metrics[number][k],np.asarray(jax.device_get(v))) for k,v in metrics.items()}
        result['full_state2_comparison'] = compare_snapshot(nnx.state(bundle).to_pure_dict(),evidence/'reference_state2')
        checkpoint = evidence/'candidate_step_000002'
        saved = save(bundle,2,checkpoint,metadata)
        result['checkpoint'] = {'path':str(checkpoint), 'manifest_sha256':sha(checkpoint/'manifest.json'),
                                 'bytes':saved['bytes'], 'save_and_hash_wall_s':saved['save_and_hash_wall_s']}
        before_bad = parameter_digest(nnx.state(bundle))
        bad = {**batch,'rewards':jnp.full_like(batch['rewards'],jnp.nan)}
        rejected = jax.block_until_ready(fn(bad))
        assert not bool(rejected['update_applied']) and parameter_digest(nnx.state(bundle)) == before_bad
        result['invalid_transaction_complete_rollback'] = True
        batch = jax.device_put(jax.tree.map(jnp.asarray,host[3]),layout)
        infer = bind_inference()
        result['inference2_comparison'] = compare_snapshot(jax.block_until_ready(infer(bundle.critic,batch['obs'],batch['actions'],batch['action_mask'])),evidence/'reference_inference2')
        result['action_gradient2_comparison'] = compare_snapshot(jax.block_until_ready(action_gradient(bundle.critic,batch)),evidence/'reference_action_gradient2')
        metrics = jax.block_until_ready(fn(batch))
        result['metrics_comparisons']['3'] = {k:compare_array(reference_metrics[3][k],np.asarray(jax.device_get(v))) for k,v in metrics.items()}
        result['full_state3_comparison'] = compare_snapshot(nnx.state(bundle).to_pure_dict(),evidence/'reference_state3')
        snapshot(nnx.state(bundle).to_pure_dict(),evidence/'candidate_state3')
        original_metrics = {k:np.asarray(jax.device_get(v)) for k,v in metrics.items()}
        del bundle, fn, infer, probe, metrics, rejected, bad, batch
        gc.collect(); jax.clear_caches()
        rebound = make_bundle(cfg)
        layout = replicated_layout(rebound)
        restored = restore(rebound,checkpoint,expected_metadata=metadata)
        assert restored['next_sampler_update'] == 3
        fresh_update = bind_update(rebound,cfg)  # AFTER precision, placement and restoration.
        fresh_infer = bind_inference()
        batch = jax.device_put(jax.tree.map(jnp.asarray,host[3]),layout)
        result['restored_inference_comparison'] = compare_snapshot(jax.block_until_ready(fresh_infer(rebound.critic,batch['obs'],batch['actions'],batch['action_mask'])),evidence/'reference_inference2')
        resumed = jax.block_until_ready(fresh_update(batch))
        result['restored_next_update_metrics'] = {k:compare_array(original_metrics[k],np.asarray(jax.device_get(v))) for k,v in resumed.items()}
        result['restored_next_update_state'] = compare_snapshot(nnx.state(rebound).to_pure_dict(),evidence/'candidate_state3')
        assert all(np.isfinite(np.asarray(jax.device_get(x))).all() for x in jax.tree.leaves(nnx.state(rebound)))
        result.update(passed=True, gradient_equivalence_passed=True, full_state_restore_jit_passed=True,
            compute_dtype='float32', master_dtype='float32', candidate_disposable_updates=7,
            temporary_value_probe_transactions=2,
            scientific_admission=False, all_state_components_checked=['online','target','value_adam','q_adam'])
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        if loader: loader.close()
        result['wall_s'] = time.monotonic()-started
        result['gpu_hours_allocated'] = args.devices*result['wall_s']/3600
        result['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['new_evidence_bytes'] = sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file())
        atomic(destination/('complete.json' if result['passed'] else 'failed.json'),result)
        atomic(destination/'lifecycle.json',result)
        print(json.dumps({'run_id':args.run_id,'passed':result['passed'],'wall_s':result['wall_s'],
                          'scientific_critic_updates':0,'canonical_admission_published':False}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['status','validate'],default='status')
    parser.add_argument('--method',choices=['scalar','divl'])
    parser.add_argument('--run-id')
    parser.add_argument('--devices',type=int,choices=[1,2,4,8],default=1)
    args = parser.parse_args()
    if args.mode == 'validate': assert args.method is not None
    run(args)


if __name__ == '__main__': main()
