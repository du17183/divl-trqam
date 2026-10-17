"""Actual CPU data/interface/protocol checks, no model updates or GPU allocation."""
from mixed_common_v2 import *
import ast,numpy as np,traceback

def main():
 result={'passed':False,'utc':stamp(),'scientific_updates':0,'GPU_launches':0,'signals_sent':0}
 try:
  assert os.environ['JAX_PLATFORMS']=='cpu' and not os.environ['CUDA_VISIBLE_DEVICES'];verify_sources();protected_audit()
  proof=read(EXP/'provenance/mixed_actor_sources010.json')
  for f,h in proof['files'].items():assert sha(EXP/f)==h,f
  from mixed_replay_v1 import MixedReplay
  from pilot8_critic_v1 import Replay
  from pilot8_actor_v5 import BCData,get_batch,checked_bc_actions
  from openpi.models.tokenizer import PaligemmaTokenizer
  from openpi.models.divl_tokenizer import DIVLTokenizer
  from critic_model_v1 import encoder_config
  loader=MixedReplay();original=Replay('train',verify_files=True);bc=BCData(original);tok=PaligemmaTokenizer(200);ctok=DIVLTokenizer(encoder_config());batches=[]
  assert len(original.by_id)==240 and sum(len(x) for x in bc.expert.values())==50
  success_indexed=sum(len(x) for x in bc.rollout.values());assert success_indexed==109
  for n in [1,2,100,101,250,251,500,501]:
   host,idx=get_batch(loader,bc,tok,ctok,n,32);assert host['bcactions'].shape==(32,10,32) and host['obs']['state'].shape==(32,8);assert np.isfinite(host['bcactions']).all() and np.all(host['bcactions'][:,:,7:]==0)
   assert len(idx['bc_indices'])==32 and all(x['source'] in ['legal_train_expert','successful_rollout'] for x in idx['bc_indices'])
   for x in idx['bc_indices']:
    if x['source']=='successful_rollout':assert original.by_id[x['episode_id']]['success'];checked_bc_actions(original.get(x['episode_id'],x['chunk_index']))
   batches.append({'update':n,'main_replay_sha256':idx['replay_index_sha256'],'bc_indices_sha256':idx['bc_index_sha256']})
  original.close();loader.close()
  c=read(EXP/'configs/checkpoint_storage011.json')
  assert len(c['catalog'])==19
  for r in c['catalog']:
   p=Path(r['logical']);assert str(p.resolve())==r['actual'];assert any(p.resolve().is_relative_to(Path(x).resolve()) for x in c['allowed_roots'])
  def mainbody(name):
   return next(x.body for x in ast.parse(Path(name).read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='main')
  old=mainbody(OLD/'code/pilot8_eval_shard_v1.py');new=mainbody(CODE/'mixed_eval_shard_v1.py')
  assert ast.dump(ast.Module(body=old,type_ignores=[]))==ast.dump(ast.Module(body=new[4:],type_ignores=[])), 'Native evaluation action/reward/controller/seed code changed'
  from mixed_eval_audit_v1 import completed_shards
  base=read(OLD/'configs/evaluation_protocol_original001.json');assert len(base['tasks'])==10 and all(len(x['episodes'])==50 for x in base['tasks'])
  for t,task in enumerate(base['tasks']):assert len({r['state_source_id'] for r in task['episodes']})==50 and len({r['official_index'] for r in task['episodes']})==50
  for m in 'BCDEFGH':
   e=read(OLD/f'configs/actor_execution_{m}001.json');p=OLD/'metrics/actor'/e['gates'][m]/'result.json';assert e['passed'] and e['devices_by_method'][m]==4 and sha(p)==e['gate_result_sha256'] and read(p)['passed']
  result.update(passed=True,batches=batches,main_indexed=290,BC_experts=50,BC_success_rollout=success_indexed,BC_failures_used=0,native_evaluation_main_AST_identical_after_provenance_guard=True,checkpoints_routed=19,authority_sha256=sha(EXP/'provenance/user_request001.txt'),actor_sourceproof_sha256=sha(EXP/'provenance/mixed_actor_sources010.json'))
 except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
 write(EXP/'metrics/mixed_queue_cpu_preflight016.json',result,True);print(json.dumps(result));sys.exit(0 if result['passed'] else 1)
if __name__=='__main__':main()
