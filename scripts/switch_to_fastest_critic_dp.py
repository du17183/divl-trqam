# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Safely hand off only our jobs to a verified matched execution configuration."""
from pathlib import Path
from datetime import datetime,timezone
import os,signal,subprocess,time,json,shutil,hashlib

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
def read(p):return json.loads(p.read_text())
def alive(pid):
    p=Path(f'/proc/{pid}/stat');return p.exists() and p.read_text().split()[2]!='Z'
def own_stop(pid,script):
    if not alive(pid):return
    p=Path(f'/proc/{pid}')
    assert str(R/'scripts'/script) in p.joinpath('cmdline').read_bytes().decode().replace('\0',' ')
    assert p.joinpath('cwd').resolve()==R.resolve()
    os.kill(pid,signal.SIGTERM)
    for _ in range(40):
        if not alive(pid):return
        time.sleep(.25)
    raise RuntimeError(f'Owned process {pid} did not exit; no force killing')
def main():
    deadline=time.monotonic()+1800
    while not (O/'critic_dp_benchmark_status.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Benchmark deadline; keep existing jobs')
        time.sleep(5)
    assert read(O/'prefetch_batch_contract.json')['passed']
    assert read(O/'portable_checkpoint_roundtrip.json')['passed']
    algebra=read(O/'gspmd_global_update_audit.json');assert algebra['passed'],'Global update algebra must pass before a multi-card switch'
    benchmark={m:read(O/f'critic_dp_benchmark_{m}.json') for m in ['scalar','divl']}
    by_count={m:{x['device_count']:x for x in v['results']} for m,v in benchmark.items()}
    assert all(set(v)=={1,2,4} for v in by_count.values()),'Incomplete real device benchmark; no restart'
    verified={m:{v['devices']:v['passed'] for v in rows} for m,rows in algebra['results'].items()}
    # BF16 kernel tiling / reduction ordering is not bitwise invariant across
    # GPU counts. Keep those failed strict numerical comparisons visible.
    # Admission uses the ORIGINAL update's full-FP32 global-algebra check,
    # identical sample/global-budget contract, and finite full-model updates.
    eligible=[n for n in [1,2,4] if all(verified[m][n] and by_count[m][n]['first_metrics']['update_applied']==1. for m in ['scalar','divl'])]
    # Both arms receive the same GPU count. Minimize time until BOTH finish.
    selected=min(eligible,key=lambda n:max(v[n]['median_update_seconds'] for v in by_count.values()))
    group={'scalar':'1,0,2,3','divl':'7,6,5,4'}
    allocations={m:','.join(g.split(',')[:selected]) for m,g in group.items()}
    protocol_paths=[R/'data/action_sensitive_replay_v2/manifest.json',R/'data/pilot_libero_v1/manifest.json',R/'configs/splits/libero_factorial_v1.json']
    immutable={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protocol_paths}
    decision=dict(selected_devices_per_method=selected,total_active_training_gpus=2*selected,
                  global_batch_per_method=256,microbatch_per_device=256//selected,allocations=allocations,
                  benchmark=benchmark,selection='Same device count for both arms, fastest measured paired completion; fixed global batch256',
                  global_update_algebra=algebra,
                  full_bf16_same_as_single_card=False,
                  numerical_note='Strict full-backbone single-vs-multi first-step comparisons did not pass. BF16/TF32 GEMM tiling and summation change numerical trajectories. Full-FP32 ORIGINAL-update algebra audit must pass; both arms restart on the SAME selected mesh. Do not interpret single-card vs multi-card results as a controlled algorithm comparison.',
                  shared_other_jobs='/home/ghz/dvg-coverage; never stopped',actor_updated=False,
                  immutable_manifest_hashes=immutable,utc=datetime.now(timezone.utc).isoformat())
    (R/'results').mkdir(exist_ok=True)
    (R/'results/critic_data_parallel_benchmark.json').write_text(json.dumps(decision,indent=2))
    print('THROUGHPUT_SELECTED',selected,allocations,flush=True)
    if selected==1:
        decision['restart']=False;decision['note']='No multi-card speedup demonstrated; original jobs continue'
        (O/'critic_dp_switch.json').write_text(json.dumps(decision,indent=2));return
    # Benchmark wrapper has resumed old processes. Stop coordinator first,
    # then each verified owned training process; retain all earlier work.
    for pid,script in [(3348531,'run_replay_v2_pipeline.py'),(3746679,'train_critic_replay_v2.py'),(3746680,'train_critic_replay_v2.py')]:own_stop(pid,script)
    archive=R/'backups/throughput_20261007/single_gpu_attempt';assert not archive.exists();archive.mkdir(parents=True)
    for path in [R/'checkpoints/critic_replay_v2',O/'pipeline_status.json',O/'pipeline_dispatch.json',O/'v2_training_runtime.json',
                 *[O/f'critic_{m}_v2_progress.json' for m in ['scalar','divl']],
                 *[O/f'critic_{m}_v2_complete.json' for m in ['scalar','divl']],
                 R/'logs/stage_v1/replay_v2_pipeline.log',
                 *[R/'logs/stage_v1'/f'critic_{m}_v2.log' for m in ['scalar','divl']],
                 *[R/'reports'/f'{m}_v2_critic_report.md' for m in ['scalar','divl']]]:
        if path.exists():
            # Explicit paths only; enforce all moves stay inside this project.
            path.resolve().relative_to(R.resolve());destination=archive/path.relative_to(R);destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(destination))
    for p in protocol_paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==immutable[str(p)]
    # No partial single-card checkpoint selection: same fresh init/order/budget
    # for both arms, with old metrics preserved as an interrupted attempt.
    with (R/'logs/stage_v1/replay_v2_pipeline.log').open('w') as f:
        process=subprocess.Popen(['/usr/bin/python3','-u',str(R/'scripts/run_replay_v2_pipeline.py'),
                                  '--reuse-completed-data-stage','--scalar-gpus',allocations['scalar'],'--divl-gpus',allocations['divl']],
                                  cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    decision.update(restart=True,new_coordinator_pid=process.pid,archive=str(archive),
                    restart_policy='Matched fresh initialization and same total2000 formal updates; old partial attempt preserved, no best checkpoint choice')
    (O/'pipeline_dispatch.json').write_text(json.dumps(dict(pid=process.pid,allocations=allocations,real_data_parallel=True,actor_frozen=True),indent=2))
    (O/'critic_dp_switch.json').write_text(json.dumps(decision,indent=2))
    (R/'results/critic_data_parallel_benchmark.json').write_text(json.dumps(decision,indent=2))
    print('REAL_DP_PIPELINE_DISPATCHED',process.pid,flush=True)
if __name__=='__main__':main()
