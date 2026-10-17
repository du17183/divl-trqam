"""One isolated real Base/Train batch throughput configuration, no scientific SFT retention."""
from pathlib import Path
import argparse
import collections
import datetime
import json
import os
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--global-batch', type=int, required=True)
    parser.add_argument('--micro-batch', type=int, required=True)
    parser.add_argument('--fsdp', type=int, required=True)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--prefetch', type=int, default=0)
    parser.add_argument('--pinned', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--stable', type=int, default=5)
    parser.add_argument('--save-checkpoint', action='store_true')
    args = parser.parse_args()
    assert args.global_batch % args.micro_batch == 0
    result_path = ROOT / 'metrics/throughput_v4' / (args.id + '.json')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    assert not result_path.exists()
    started = time.monotonic()
    result = {'schema': 'actual_real_batch_training_throughput_v4', 'config': vars(args), 'passed': False,
              'pid': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'steps': [], 'gpu_samples': [],
              'scope': 'real admitted Train observations/actions and pi05_base; disposable optimizer updates; not scientific SFT',
              'flow_law': 'native noise/time/augmentation; real7/full executed H10 mask; one optimizer update after global gradient accumulation',
              'gpu_communication_s': None, 'gpu_communication_reason': 'not separated from native training wall by CUDA/NCCL events',
              'rematerialization': 'native Gemma block and SigLIP remat unchanged', 'bf16_compute_fp32_master': False, 'fp32_compute_fp32_master': True, 'bf16_accumulation_gate': 'rejected 7.94% relative gradient difference; threshold unchanged'}
    stop_event = threading.Event()
    loader = None
    physical_ids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    def sample_gpu():
        while not stop_event.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], text=True)
                for line in raw.strip().splitlines():
                    fields = [x.strip() for x in line.split(',')]
                    if int(fields[0]) in physical_ids:
                        result['gpu_samples'].append({'monotonic_offset_s': time.monotonic() - started,
                                                      'gpu': int(fields[0]), 'memory_used_mib': float(fields[1]),
                                                      'utilization_percent': float(fields[2]), 'power_w': float(fields[3])})
            except Exception as error: result['gpu_sample_error'] = str(error)
            stop_event.wait(1)
    thread = threading.Thread(target=sample_gpu, daemon=True); thread.start()
    try:
        assert json.loads((ROOT / 'metrics/fm_native_gate004.json').read_text())['passed']
        import jax
        import jax.numpy as jnp
        import numpy as np
        import orbax.checkpoint as ocp
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.training import sharding
        from benchmark_data_v2 import NativeExpertLoader
        from benchmark_fm_v3 import as_batch, make_config, native_training_module, make_step
        assert len(jax.devices()) == len(physical_ids)
        assert args.micro_batch % len(physical_ids) == 0
        jax.config.update('jax_default_matmul_precision', 'highest')
        jax.config.update('jax_compilation_cache_dir', str(ROOT / 'cache/jax'))
        loader = NativeExpertLoader(10, 20261009, workers=args.workers, prefetch=args.prefetch, pinned=args.pinned)
        loader.set_tokens(PaligemmaTokenizer(200))
        result['train_manifest_sha256'] = loader.manifest_sha
        cfg = make_config(args.global_batch, args.fsdp)
        mesh = sharding.make_mesh(args.fsdp)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        before = time.monotonic()
        state, state_sharding = native_training_module().init_train_state(cfg, jax.random.key(0), mesh, resume=False)
        jax.block_until_ready(state)
        result['base_load_s'] = time.monotonic() - before
        dtype_counts = collections.Counter(str(x.dtype) for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter)))
        assert set(dtype_counts) == {'float32'}
        result['master_dtypes'] = dict(dtype_counts)
        fn = jax.jit(make_step(cfg, args.micro_batch), in_shardings=(replicated, state_sharding, data_sharding),
                     out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        rng = jax.random.key(20261009)
        total_steps = 1 + args.warmup + args.stable
        stable_start = None
        for index in range(total_steps):
            if index == 1 + args.warmup: stable_start = time.monotonic()
            data, sources, wait_s, read_s = loader.next(args.global_batch)
            before = time.monotonic()
            batch = jax.device_put(as_batch(data), data_sharding)
            jax.block_until_ready(batch)
            transfer_s = time.monotonic() - before
            before = time.monotonic()
            with sharding.set_mesh(mesh): state, stats = jax.block_until_ready(fn(rng, state, batch))
            compute_s = time.monotonic() - before
            values = {k: float(x) for k, x in stats.items()}
            assert all(np.isfinite(x) for x in values.values()) and values['update_norm'] > 0
            kind = 'compile' if index == 0 else 'warmup' if index <= args.warmup else 'stable'
            row = {'index': index, 'kind': kind, 'loader_wait_s': wait_s, 'loader_active_read_s': read_s,
                   'transfer_s': transfer_s, 'compute_s': compute_s, **values}
            result['steps'].append(row)
            if index == 0:
                result['source_first_global_batch'] = sources
                result['compile_plus_first_update_s'] = compute_s
                try: result['compiled_memory_analysis'] = str(fn.lower(rng, state, batch).compile().memory_analysis())
                except Exception as error: result['compiled_memory_analysis'] = str(error)
            print('REAL_PROFILE_UPDATE', args.id, row, flush=True)
        stable_wall = time.monotonic() - stable_start
        stable_rows = [x for x in result['steps'] if x['kind'] == 'stable']
        result.update(stable_wall_s=stable_wall, samples_per_s=args.global_batch * args.stable / stable_wall,
                      updates_per_s=args.stable / stable_wall, global_batch=args.global_batch,
                      micro_batch_global=args.micro_batch, micro_batch_per_gpu=args.micro_batch // len(physical_ids),
                      gradient_accumulation_steps=args.global_batch // args.micro_batch,
                      stable_compute_s=sum(x['compute_s'] for x in stable_rows),
                      stable_loader_wait_s=sum(x['loader_wait_s'] for x in stable_rows),
                      stable_loader_active_read_s=sum(x['loader_active_read_s'] for x in stable_rows),
                      stable_transfer_s=sum(x['transfer_s'] for x in stable_rows))
        if args.save_checkpoint:
            before = time.monotonic()
            path = ROOT / 'checkpoints/throughput_v4' / args.id / 'full_train_state'
            with ocp.PyTreeCheckpointer() as ck:
                # Static model_def/tx reconstructed by code; actual master params + optimizer moments + update index saved.
                ck.save(str(path.resolve()), {'params': state.params.to_pure_dict(), 'optimizer': state.opt_state, 'step': state.step})
            result['full_checkpoint_save_s'] = time.monotonic() - before
            result['full_checkpoint_bytes'] = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        else:
            result['full_checkpoint_save_s'] = None
            result['checkpoint_measurement_scope'] = 'measured in dedicated confirmation configuration only'
        result['peak_total_device_memory_mib'] = max((x['memory_used_mib'] for x in result['gpu_samples']), default=None)
        result['gpu_utilization_samples_mean_percent'] = float(np.mean([x['utilization_percent'] for x in result['gpu_samples']]))
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        if loader: loader.close()
        stop_event.set(); thread.join(timeout=3)
        result['wall_s'] = time.monotonic() - started
        result['gpu_hours_allocated'] = len(physical_ids) * result['wall_s'] / 3600
        result['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with result_path.open('x') as f: f.write(json.dumps(result, indent=2) + '\n')
        print('REAL_PROFILE_COMPLETE', args.id, result.get('samples_per_s'), result['passed'], flush=True)


if __name__ == '__main__': main()
