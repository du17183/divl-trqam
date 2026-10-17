# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Resource-aware actual real-batch 1/2/4/8GPU throughput search after native engineering gates."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/opt/divl-trqam/.venv_actor/bin/python'
PAUSED_OLD_GPU_PIDS = {2232947, 2232945, 2455901, 2455900}


def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def env_for(gpus):
    runtime = json.loads((ROOT / 'configs/runtime_libraries_v1.json').read_text())
    return {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
            'LD_LIBRARY_PATH': ':'.join(runtime['library_paths']), 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)),
            'OPENPI_DATA_HOME': str(ROOT / 'cache/openpi'),
            'PYTHONPATH': ':'.join(str(ROOT / x) for x in ['vendor/openpi/src', 'vendor/openpi/packages/openpi-client/src', 'scripts']),
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '2', 'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false'}


def resource_gate(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for line in rows.strip().splitlines():
        index, uuid, free, util = [v.strip() for v in line.split(',')]
        devices[uuid] = {'gpu': int(index), 'free_mib': float(free), 'utilization_percent': float(util)}
    evidence = []
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    for line in compute.strip().splitlines():
        if not line.strip(): continue
        uuid, pid, memory = [x.strip() for x in line.split(',')]
        if devices[uuid]['gpu'] not in gpus: continue
        p = Path('/proc') / pid
        try:
            argv = p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
            status = p.joinpath('status').read_text()
        except FileNotFoundError: continue
        state_line = next(x for x in status.splitlines() if x.startswith('State:'))
        allowed = int(pid) in PAUSED_OLD_GPU_PIDS and 'privileged_rl_validation' in argv and 'T (stopped)' in state_line
        evidence.append({'gpu': devices[uuid]['gpu'], 'pid': int(pid), 'full_argv': argv, 'state': state_line, 'allowed_old_paused': allowed})
        if not allowed: return False, {'reason': 'another compute process holds selected GPU', 'processes': evidence, 'devices': devices}
    for d in devices.values():
        if d['gpu'] in gpus and (d['free_mib'] < 230000 or d['utilization_percent'] > 5):
            return False, {'reason': 'selected GPU not idle or memory margin insufficient', 'devices': devices, 'processes': evidence}
    return True, {'devices': devices, 'processes': evidence, 'checked_utc': stamp()}


def wait_path(path, state, phase):
    state['stage'] = phase; atomic(ROOT / 'metrics/throughput_lifecycle_v4.json', state)
    while not path.exists(): time.sleep(15)
    value = json.loads(path.read_text())
    assert value['passed'], str(path) + ' did not pass; stop downstream'
    return value


def run_job(config, state):
    gpus = config['gpus']
    while True:
        passed, evidence = resource_gate(gpus)
        if passed: break
        state.update(stage='waiting_resources', resource_wait=evidence)
        atomic(ROOT / 'metrics/throughput_lifecycle_v4.json', state)
        time.sleep(30)
    argv = [PYTHON, '-u', str(ROOT / 'scripts/profile_fm_v4.py'), '--id', config['id'], '--global-batch', str(config['global_batch']),
            '--micro-batch', str(config['micro_batch']), '--fsdp', str(len(gpus)), '--workers', str(config['workers']),
            '--prefetch', str(config['prefetch'])]
    if config.get('pinned'): argv.append('--pinned')
    if config.get('save_checkpoint'): argv.append('--save-checkpoint')
    log_path = ROOT / 'logs' / (config['id'] + '.log')
    with log_path.open('x') as log:
        p = subprocess.Popen(argv, cwd=ROOT, env=env_for(gpus), stdout=log, stderr=subprocess.STDOUT)
        active = {'pid': p.pid, 'argv': argv, 'config': config, 'resource_preflight': evidence,
                  'birth_ticks': Path('/proc', str(p.pid), 'stat').read_text().split()[21], 'started_utc': stamp()}
        state.update(stage='real_training_throughput', active=active)
        atomic(ROOT / 'metrics/throughput_lifecycle_v4.json', state)
        code = p.wait()
    outcome_path = ROOT / 'metrics/throughput_v4' / (config['id'] + '.json')
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {'passed': False, 'error': log_path.read_text()[-16000:]}
    finished = {**active, 'exit_code': code, 'ended_utc': stamp(), 'result': str(outcome_path),
                'passed': outcome['passed'], 'samples_per_s': outcome.get('samples_per_s')}
    state['finished'].append(finished); state['active'] = None
    atomic(ROOT / 'metrics/throughput_lifecycle_v4.json', state)
    if code != 0 or not outcome['passed']:
        error = outcome.get('error', '')
        if any(x in error.lower() for x in ['out of memory', 'resource_exhausted', 'cuda_error_out_of_memory']):
            finished['status'] = 'INFEASIBLE_OOM_NO_SCIENTIFIC_TRAINING'
            return None
        raise RuntimeError('Actual native throughput engineering failed in %s; preserve logs before technical repair: %s' % (config['id'], error[-3000:]))
    return outcome


def main():
    state = {'schema': 'native_real_throughput_supervisor_v1', 'supervisor_pid': os.getpid(), 'started_utc': stamp(),
             'stage': 'waiting_native_gate', 'active': None, 'finished': [], 'scientific_training_started': False}
    try:
        wait_path(ROOT / 'metrics/fm_native_gate004.json', state, 'waiting_native_gate')
        state['stage'] = 'loader_equivalence'; atomic(ROOT / 'metrics/throughput_lifecycle_v4.json', state)
        with (ROOT / 'logs/loader_equivalence001.log').open('x') as log:
            p = subprocess.Popen([PYTHON, '-u', str(ROOT / 'scripts/loader_equivalence_v1.py')], env=env_for([]), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            assert p.wait() == 0, 'Real loader equivalence failed'
        wait_path(ROOT / 'metrics/loader_equivalence001.json', state, 'loader_equivalence')
        grid = []
        for count, gpus in [(1, [6]), (2, [6, 7]), (4, [4, 5, 6, 7]), (8, list(range(8)))]:
            for per_gpu in [1, 2, 4, 8, 16, 32, 64]:
                micro = per_gpu * count
                if micro > 64: continue
                grid.append({'id': 'fm_profile_g%d_m%d_b64' % (count, per_gpu), 'gpus': gpus,
                             'global_batch': 64, 'micro_batch': micro, 'workers': 4, 'prefetch': 2, 'pinned': False})
        manifest = {'scope': 'actual native pi05_base and admitted Train examples', 'baseline_global_batch': 64, 'grid': grid,
                    'expansion': 'best stable primary GPU layout tested at global128/256/512 with per-GPU micro64 maximum; stop OOM ladder',
                    'confirmation': 'best stable configuration with workers0/4/8, prefetch0/2 and pinned host-memory variant; actual full checkpoint save',
                    'stable_selection': 'largest measured end-to-end samples/s among finite stable configurations, initial native contract/equivalence gates mandatory',
                    'all_scientific_arms': 'freeze selected Global Batch once; identical update and sample budgets',
                    'no_saved_profile_weights_as_sft_init': True, 'created_utc': stamp()}
        with (ROOT / 'configs/throughput_search_v4.json').open('x') as f: f.write(json.dumps(manifest, indent=2) + '\n')
        outcomes = []
        for config in grid:
            # An OOM at a fixed device count closes larger micro-batches at that count.
            oom_counts = {len(x['config']['gpus']) for x in state['finished'] if x.get('status') == 'INFEASIBLE_OOM_NO_SCIENTIFIC_TRAINING'}
            if len(config['gpus']) in oom_counts:
                state.setdefault('not_run', []).append({'config': config, 'reason': 'higher microbatch after OOM; avoid sustained OOM'})
                continue
            outcome = run_job(config, state)
            if outcome: outcomes.append((config, outcome))
        assert outcomes
        best_config, best = max(outcomes, key=lambda x: x[1]['samples_per_s'])
        for global_batch in [128, 256, 512]:
            config = {**best_config, 'id': 'fm_profile_expand_b%d' % global_batch, 'global_batch': global_batch,
                      'micro_batch': min(global_batch, 64 * len(best_config['gpus']))}
            outcome = run_job(config, state)
            if outcome is None: break
            outcomes.append((config, outcome))
        best_config, best = max(outcomes, key=lambda x: x[1]['samples_per_s'])
        for workers, prefetch, pinned in [(0, 0, False), (4, 0, False), (8, 2, False), (4, 2, True)]:
            config = {**best_config, 'id': 'fm_profile_loader_w%d_p%d_pin%d' % (workers, prefetch, pinned),
                      'workers': workers, 'prefetch': prefetch, 'pinned': pinned}
            outcome = run_job(config, state)
            if outcome: outcomes.append((config, outcome))
        best_config, best = max(outcomes, key=lambda x: x[1]['samples_per_s'])
        confirmed = {**best_config, 'id': 'fm_profile_selected_confirmation', 'save_checkpoint': True}
        confirmation = run_job(confirmed, state)
        assert confirmation
        selected = {'schema': 'measured_native_training_configuration_v1', 'passed': True, 'config': confirmed,
                    'measured': confirmation, 'selection_evidence': [{'config': c, 'samples_per_s': x['samples_per_s']} for c, x in outcomes],
                    'scientific_global_batch': confirmed['global_batch'], 'scientific_updates_per_s': confirmation['updates_per_s'],
                    'source_manifest_sha256': confirmation['train_manifest_sha256'], 'selected_utc': stamp(),
                    'scope': 'native masked FM SFT only; QAM/TRQAM will have independent real profiles without changing their mathematical flows'}
        atomic(ROOT / 'configs/selected_throughput_v4.json', selected)
        fields = ['id', 'gpus', 'global_batch', 'micro_batch', 'workers', 'prefetch', 'pinned', 'samples_per_s', 'updates_per_s',
                  'base_load_s', 'compile_plus_first_update_s', 'stable_loader_wait_s', 'stable_transfer_s', 'peak_total_device_memory_mib',
                  'gpu_communication_s', 'full_checkpoint_save_s', 'wall_s', 'gpu_hours_allocated']
        with (ROOT / 'metrics/sft_throughput_search.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for config, outcome in outcomes + [(confirmed, confirmation)]: writer.writerow({k: config.get(k, outcome.get(k)) for k in fields})
        lines = ['# B300 actual SFT throughput profile', '', 'Real admitted Train RGB and H10 actions were used. Every job initialized from generic pi05_base; profiling updates are disposable.', '',
                 'Selected configuration: ' + json.dumps(confirmed), '',
                 'Confirmation samples/s: %.6f; updates/s: %.6f; full checkpoint save: %.6fs.' % (confirmation['samples_per_s'], confirmation['updates_per_s'], confirmation['full_checkpoint_save_s']), '',
                 'Compile, load, transfer, data wait and allocated GPU hours are separate in the CSV/JSON. GPU communication is unknown when not directly separated by CUDA/NCCL events; device memory includes any declared old paused-process baseline. Native Gemma/SigLIP rematerialization and FP32 highest compute/FP32 master parameters are retained; BF16 accumulation was rejected by the unchanged equivalence gate. No profiling checkpoint is reused as SFT initialization.', '',
                 'Evidence: metrics/sft_throughput_search.csv, metrics/throughput_v4/*.json, per-job logs, native FM gate and real loader exact equivalence gate.', '',
                 'SFT scientific training and all ten-task checkpoint evaluation are the next stage; they are not inferred from a throughput profile.']
        for name in ['b300_throughput_profile.md', 'throughput_optimization.md', '12_throughput_optimization.md']:
            with (ROOT / 'reports' / name).open('x') as f: f.write('\n'.join(lines) + '\n')
        state.update(stage='throughput_complete_sft_pending', passed=True, ended_utc=stamp())
    except Exception as error:
        state.update(stage='STOP_ENGINEERING_GATE', passed=False, error=str(error), ended_utc=stamp())
        raise
    finally: atomic(ROOT / 'metrics/throughput_lifecycle_v4.json', state)


if __name__ == '__main__': main()
