"""Read once into immutable UUID snapshots. CPU only, no training controls."""
from mixed_common_v2 import *
import argparse,csv,uuid,collections

def csvrows(name):
 p=EXP/'metrics'/name
 if not p.exists():return []
 with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
 a=argparse.ArgumentParser();a.add_argument('--initialize-reports',action='store_true');args=a.parse_args()
 verify_sources();protected_audit()
 ready=read(EXP/'provenance/mixed_actor_queue_ready010.json')
 for rel,h in ready['files'].items():assert sha(EXP/rel)==h,rel
 state=read(EXP/'metrics/pipeline_state001.json')
 assert not any(r['name'].startswith('report_') for r in state.get('active_jobs',[])),'CPU reporter is still writing; snapshot retries later'
 if args.initialize_reports:
  assert not state.get('actor_updates') and not (EXP/'metrics/all_metrics.json').exists()
  r=subprocess.run([ACTOR,'-u',str(CODE/'mixed_results_report_v1.py')],cwd=ROOT,env=env(),capture_output=True,text=True)
  assert r.returncode==0,(r.stdout,r.stderr)
 assert (EXP/'metrics/all_metrics.json').exists()
 data=read(EXP/'metrics/all_metrics.json')
 expected={k for k,v in state.get('evaluations',{}).items() if v['status']=='AUDITED_COMPLETE500'}
 reported={r['evaluation_id'] for r in data['results'] if r['episodes']==500}
 assert expected==reported,'Latest admitted500 is not reported yet; snapshot retries later'
 for rel,h in data['actual_critic_sampling']['files'].items():assert sha(EXP/'metrics'/rel)==h,rel
 source=[*sorted((EXP/'reports').glob('*.md')),*sorted((EXP/'metrics').glob('*.csv'))]
 source +=[EXP/'metrics'/n for n in ['all_metrics.json','mixed_latest_snapshot001.json','mixed_sampling_summary018.json','expert_h5_terminal_coverage013.json','mixed_queue_cpu_preflight017.json','mixed_queue_cpu_gate018.json','pipeline_state001.json'] if (EXP/'metrics'/n).exists()]
 source +=[EXP/'provenance'/n for n in ['user_request001.txt','mixed_actor_queue_ready010.json','mixed_actor_sources010.json','mixed_diagnostics_sources015.json','mixed_queue_engineering018.json']]
 source +=[EXP/'configs'/n for n in ['protocol001.json','checkpoint_storage011.json','mixed_physical_validation015.json']]
 source +=[EXP/'data/performance2/mixed_manifest001.json']
 if (EXP/'metrics/mixed_critic_comparison_complete015.json').exists():source.append(EXP/'metrics/mixed_critic_comparison_complete015.json')
 destination=EXP/'provenance/artifact_snapshot019'/uuid.uuid4().hex;destination.mkdir(parents=True,exist_ok=False);files=[]
 for p in source:
  b=p.read_bytes();rel=p.relative_to(EXP).as_posix();q=destination/rel;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  files.append({'relative':rel,'bytes':len(b),'sha256':sha(q)})
 # Use accepted original metric records as separate historical inputs.
 legacy=read(OLD/'metrics/all_metrics.json');assert legacy['complete']
 historical={'methods':legacy['methods'],'per_task':legacy['per_task'],'paired':legacy.get('paired',legacy.get('comparisons',[])),'source_sha256':sha(OLD/'metrics/all_metrics.json'),'source':str(OLD/'metrics/all_metrics.json')}
 write(destination/'metrics/historical_results.json',historical,True)
 files.append({'relative':'metrics/historical_results.json','bytes':(destination/'metrics/historical_results.json').stat().st_size,'sha256':sha(destination/'metrics/historical_results.json')})
 payload={'utc':data['utc'],'captured_utc':stamp(),'data':data,'historical':historical,'state':read(destination/'metrics/pipeline_state001.json'),'latest':read(destination/'metrics/mixed_latest_snapshot001.json'),'sampler':read(destination/'metrics/mixed_sampling_summary018.json'),'protocol':read(destination/'configs/protocol001.json'),'manifest':read(destination/'data/performance2/mixed_manifest001.json'),'terminal':read(destination/'metrics/expert_h5_terminal_coverage013.json'),'critic_comparison':read(destination/'metrics/mixed_critic_comparison_complete015.json') if (destination/'metrics/mixed_critic_comparison_complete015.json').exists() else None,'csv':{p.name:list(csv.DictReader((destination/p.relative_to(EXP)).open(newline=''))) for p in source if p.suffix=='.csv' and p.name not in ['actor_actual_source_draws.csv','actor_training_updates.csv','mixed_replay_actual_sampling.csv','critic_actual_trajectory_coverage.csv']},'source_files':files,'complete24000':data['complete_24000']}
 write(destination/'workbook_input.json',payload,True);files.append({'relative':'workbook_input.json','bytes':(destination/'workbook_input.json').stat().st_size,'sha256':sha(destination/'workbook_input.json')})
 receipt={'utc':stamp(),'snapshot_path':str(destination),'files':files,'complete24000':data['complete_24000'],'reported500':sorted(reported),'scope':'Immutable bytes, exact SHA, original historical results separate; progress files may advance during capture interval','no_GPU_or_scientific_updates':True}
 write(destination/'snapshot_receipt.json',receipt,True);print(json.dumps(receipt))

if __name__=='__main__':main()
