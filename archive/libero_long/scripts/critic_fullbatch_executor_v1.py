# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Gated complete-batch profile/training candidate. No actor, QAM, or auto-launch."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_resources(gpus):
    # Reuse the existing read-only check, then strengthen paused-PID birth/cwd checks.
    from run_profile_v4 import resource_gate
    ok,evidence=resource_gate(gpus)
    assert ok,evidence
    original={2232947:'1025642706',2232945:'1025642706',
              2455901:'1025753435',2455900:'1025753435'}
    for row in evidence.get('processes',[]):
        p=Path('/proc')/str(row['pid'])
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        assert row['pid'] in original and s[19]==original[row['pid']] and s[0]=='T'
        assert (p/'cwd').resolve()==Path('/home/researcher/privileged_rl_validation').resolve()
    return evidence


def run(args):
    from critic_state_v1 import readiness, atomic, save, restore, digest
    if args.mode=='status':
        print(json.dumps({'readiness':readiness(require_execution=False),
           'scientific_readiness':readiness(require_execution=True),
           'candidate_only':True,'gpu_initialized':False,'workers_launched':False}))
        return
    assert args.run_id and args.run_id.replace('_','').replace('-','').isalnum()
    assert args.method in ['scalar','divl']
    assert args.devices in [1,2,4,8]
    assert args.warmup>=1 and args.stable>=2
    from critic_fullbatch_v1 import strict_upstream
    prereq=strict_upstream(scientific=args.mode=='train')
    protocol=prereq['protocol']
    assert protocol['global_batch']==256 and protocol['budget_updates']==2000
    gpus=[int(v) for v in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(gpus)==len(set(gpus))==args.devices
    if args.mode=='train':
        assert prereq['execution']['device_count']==args.devices
    resource_evidence=validate_resources(gpus)
    destination=ROOT/('experiments/critic' if args.mode=='train' else 'metrics/critic_fullbatch_profiles')/args.run_id
    destination.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    record={'schema':'native_fullbatch_executor_candidate001','mode':args.mode,'method':args.method,
      'run_id':args.run_id,'pid':os.getpid(),'started_utc':stamp(),'prerequisites':prereq,
      'passed':False,'scientific_critic':args.mode=='train','native_full_batch_only':True,
      'global_batch':256,'micro_batch_global':256,'gradient_accumulation_steps':1,
      'devices':gpus,'resource_preflight':resource_evidence,'updates':0,
      'gpu_communication_s':None,'gpu_communication_reason':'not separately instrumented',
      'data_pipeline':'audited lazy RGB reads; prefetch optimization not yet benchmarked',
      'scope':'real full public pretrained VLM and selected-policy Replay if upstream passes',
      'physics_ranking_admitted':False,'actor_updates':0,'gpu_samples':[],'cpu_samples':[]}
    stop=threading.Event()
    sampler=None
    loader=None
    def telemetry():
        last=None
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw',
                                              '--format=csv,noheader,nounits'],text=True)
                for line in raw.strip().splitlines():
                    i,m,u,p=[v.strip() for v in line.split(',')]
                    if int(i) in gpus:
                        record['gpu_samples'].append({'offset_s':time.monotonic()-started,'gpu':int(i),
                           'total_device_memory_mib':float(m),'util_percent':float(u),'power_w':float(p)})
                stats=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
                total=sum(stats); idle=stats[3]+stats[4]
                if last and total>last[0]:
                    record['cpu_samples'].append({'offset_s':time.monotonic()-started,
                     'system_busy_percent':100*(1-(idle-last[1])/(total-last[0]))})
                last=(total,idle)
            except Exception as error:
                record['telemetry_error']=repr(error)
            stop.wait(2)
    try:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from critic_model_v1 import CriticConfig,encoder_config
        from critic_replay_v1 import AuditedReplay,collate
        from critic_fullbatch_v1 import make_bundle,bind_update,replicated_layout,parameter_digest
        from openpi.models.divl_tokenizer import DIVLTokenizer
        assert len(jax.devices())==args.devices and all(d.platform=='gpu' for d in jax.devices())
        jax.config.update('jax_default_matmul_precision','highest')
        jax.config.update('jax_compilation_cache_dir',str(ROOT/'cache/jax_critic_fullbatch'))
        sampler=threading.Thread(target=telemetry,daemon=True);sampler.start()
        cfg=CriticConfig(value_type=args.method,seed=protocol['seed_pilot'],
          expectile=protocol['scalar_expectile'],tau_base=protocol['divl']['tau_base'],
          alpha=protocol['divl']['entropy_alpha'],ema_rate=protocol['target_new_weight'],
          max_grad_norm=protocol['gradient_clip'],learning_rate=protocol['learning_rate'])
        before=time.monotonic()
        bundle=make_bundle(cfg)
        data_layout=replicated_layout(bundle)
        record['initialization_and_layout_s']=time.monotonic()-before
        record['initial_q_sha256']=parameter_digest(nnx.state(bundle.critic.q_head,nnx.Param))
        record['initial_encoder_sha256']=parameter_digest(nnx.state(bundle.critic.encoder,nnx.Param))
        metadata={'value_type':args.method,'seed':cfg.seed,'scientific_critic':args.mode=='train',
          'protocol_sha256':prereq['protocol_sha256'],'replay_manifest_sha256':prereq['replay_manifest_sha256'],
          'sft_selection_sha256':prereq['sft_selection_sha256'],
          'execution_sources_sha256':prereq['execution_sources_sha256'],'native_full_batch_only':True,
          'global_batch':256,'compute_dtype':'float32','master_dtype':'float32'}
        start_update=0
        if args.resume:
            assert args.mode=='train','Do not warm-start profiles from scientific/temporary weights'
            restored=restore(bundle,args.resume,expected_metadata=metadata)
            start_update=restored['update']
            assert start_update<2000
            record['resumed_from']=str(Path(args.resume).resolve())
        # Bind only after master precision, placement and restoration are complete.
        update_fn=bind_update(bundle,cfg)
        loader=AuditedReplay(prereq['replay_manifest_path'])
        tokenizer=DIVLTokenizer(encoder_config())
        norm=json.loads((ROOT/'data/sft/common_norm_stats_v1.json').read_text())
        steps=2000 if args.mode=='train' else 1+args.warmup+args.stable
        stable_start=None
        records=[]
        with (destination/'updates.jsonl').open('x') as log:
            for number in range(start_update+1,steps+1):
                if args.mode=='profile' and number==2+args.warmup: stable_start=time.monotonic()
                before=time.monotonic()
                rows,index_sha=loader.samples(number,256,cfg.seed)
                host=collate(rows,loader.languages,tokenizer,norm)
                read_s=time.monotonic()-before
                before=time.monotonic()
                batch=jax.device_put(jax.tree.map(jnp.asarray,host),data_layout)
                jax.block_until_ready(batch)
                transfer_s=time.monotonic()-before
                before=time.monotonic()
                metrics=jax.block_until_ready(update_fn(batch))
                compute_s=time.monotonic()-before
                values={k:float(v) for k,v in metrics.items()}
                assert all(np.isfinite(v) for v in values.values()) and bool(metrics['update_applied']),values
                kind=('scientific' if args.mode=='train' else
                      'compile' if number==1 else 'warmup' if number<=1+args.warmup else 'stable')
                row={'update':number,'utc':stamp(),'kind':kind,'sample_indices_sha256':index_sha,
                     'batch':256,'read_and_collate_s':read_s,'transfer_s':transfer_s,'compute_s':compute_s,
                     'wall_s':read_s+transfer_s+compute_s,**values}
                records.append(row)
                log.write(json.dumps(row)+'\n');log.flush()
                record['updates']=number
                atomic(destination/'lifecycle.json',{k:v for k,v in record.items() if k not in ['gpu_samples','cpu_samples']})
                print('CRITIC_NATIVE_FULLBATCH',json.dumps(row),flush=True)
                if args.mode=='train' and number in protocol['checkpoints']:
                    path=ROOT/'checkpoints/critic'/args.run_id/('step_%06d'%number)
                    ck=save(bundle,number,path,metadata)
                    record.setdefault('checkpoints',[]).append({'path':str(path),'bytes':ck['bytes'],
                       'save_and_hash_wall_s':ck['save_and_hash_wall_s'],'manifest_sha256':digest(path/'manifest.json')})
        if args.mode=='profile':
            stable_wall=time.monotonic()-stable_start
            stable=[r for r in records if r['kind']=='stable']
            assert len(stable)==args.stable
            record.update(samples_per_s=256*len(stable)/stable_wall,updates_per_s=len(stable)/stable_wall,
              stable_wall_s=stable_wall,compile_plus_first_update_s=records[0]['compute_s'],
              stable_read_and_collate_s=sum(r['read_and_collate_s'] for r in stable),
              stable_transfer_s=sum(r['transfer_s'] for r in stable))
            path=ROOT/'data/gates/critic_fullbatch_profiles'/args.run_id/('step_%06d'%steps)
            ck=save(bundle,steps,path,metadata)
            record['temporary_checkpoint']={'path':str(path),'bytes':ck['bytes'],
               'save_and_hash_wall_s':ck['save_and_hash_wall_s'],'not_scientific_initialization':True}
            # Profiling alone DOES NOT admit GPU layout or create the scientific execution config.
            record['gradient_equivalence_admitted']=False
            record['full_pretrained_gate_passed']=False
        record['passed']=True
    except Exception:
        record['error']=traceback.format_exc()
        raise
    finally:
        if loader: loader.close()
        stop.set()
        if sampler: sampler.join(timeout=3)
        record['wall_s']=time.monotonic()-started
        record['gpu_hours_allocated']=args.devices*record['wall_s']/3600
        record['completed_utc']=stamp()
        if record['gpu_samples']:
            record['peak_total_device_memory_mib']=max(r['total_device_memory_mib'] for r in record['gpu_samples'])
            record['mean_gpu_utilization_percent']=sum(r['util_percent'] for r in record['gpu_samples'])/len(record['gpu_samples'])
        atomic(destination/('complete.json' if record['passed'] else 'failed.json'),record)
        atomic(destination/'lifecycle.json',record)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['status','profile','train'],default='status')
    p.add_argument('--method',choices=['scalar','divl'])
    p.add_argument('--run-id')
    p.add_argument('--devices',type=int,default=1)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--stable',type=int,default=5)
    p.add_argument('--resume')
    run(p.parse_args())


if __name__=='__main__': main()
