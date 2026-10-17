"""CPU-only monitoring: verify hashes and identities, count applied updates."""
from mixed_common_v2 import *
import collections

def logrows(path):
 if not path.exists():return []
 # A producer may be between write() and flush(); count only complete lines.
 return [json.loads(x) for x in path.read_bytes().splitlines(keepends=True) if x.endswith(b'\n') and x.strip()]

def main():
 verify_sources();protected=protected_audit();assert sha(EXP/'provenance/user_request001.txt')=='8e5ae1dd9f4c0d59f2ee94d06e933899c570398ee1abf018185389471d71a55e'
 for f,h in read(EXP/'provenance/mixed_training_sources009.json')['files'].items():assert sha(EXP/f)==h,f
 state=read(EXP/'metrics/pipeline_state001.json');live=[]
 for r in [state['coordinator'],*state.get('active_jobs',[])]:
  if not Path('/proc',str(r['pid'])).exists():live.append({'registered':r,'exists':False});continue
  live.append({'registered':r,'actual':exact(r)})
 critics={}
 for method in ['scalar','divl']:
  d=EXP/'metrics/critic'/('mixed_'+method+'_train001');p=d/'train_steps.jsonl';rows=logrows(p)
  assert [x['update'] for x in rows]==list(range(1,len(rows)+1)) and all(x['update_applied']==1 and x['batch']==128 for x in rows)
  rec={'real_updates':len(rows),'last':rows[-1] if rows else None,'lifecycle':read(d/'lifecycle.json') if (d/'lifecycle.json').exists() else None,'result':read(d/'result.json') if (d/'result.json').exists() else None,'sameconfig_gate':read(d/'same_configuration_gate.json') if (d/'same_configuration_gate.json').exists() else None};critics[method]=rec
 a,b=(critics[m] for m in ['scalar','divl']);n=min(a['real_updates'],b['real_updates']);pa=EXP/'metrics/critic/mixed_scalar_train001/train_steps.jsonl';pb=EXP/'metrics/critic/mixed_divl_train001/train_steps.jsonl'
 if n:
  aa=logrows(pa)[:n];bb=logrows(pb)[:n]
  assert all(x['sample_index_sha256']==y['sample_index_sha256'] and x['samples']==y['samples'] for x,y in zip(aa,bb,strict=True))
 out={'utc':stamp(),'stage':state['stage'],'authority_sha256':sha(EXP/'provenance/user_request001.txt'),'sampler_choice_sha256':sha(EXP/'provenance/user_sampler_choice008.txt'),'manifest_sha256':sha(EXP/'data/performance2/mixed_manifest001.json'),'experts_audited':50,'mixed_trajectories':290,'window_counts':read(EXP/'metrics/mixed_replay_cpu_preflight009.json')['source_windows'],'critic':critics,'paired_update_prefix_verified':n,'processes':live,'protected_processes_verified':len(protected),'new_actor_updates':state.get('actor_updates',{}),'new_evaluation_episodes':state.get('completed_evaluation_episodes',0),'monitor_signals_sent':0,'stage_error':state.get('error')}
 write(EXP/'metrics/mixed_latest_snapshot001.json',out)
 report=['# 03 Mixed Critic 训练进度','',f"UTC：{out['utc']}",'','|Critic|实际更新|预算|同配置恢复／fresh JIT／next update|','|---|---:|---:|---|']
 for m,v in critics.items():
  gate=v['sameconfig_gate'];passed=gate and gate['passed'];report.append(f"|{m}|{v['real_updates']}|2000|{'通过，误差0，完整digest一致' if passed else '尚未完成'}|")
 report+=['',f'已逐条核对 Scalar/DIVL 的前 {n} 次实际样本索引完全一致。', '', '主池50专家＋240Train Rollout，共22511合法起点；保留0、5、10…决策起点，不设来源Quota。各更新实际来源比例与轨迹覆盖从train_steps.jsonl读取，不将进入索引数量当成实际覆盖。','', '当前阶段没有新的Actor闭环成功率。Critic损失和恢复门槛不能替代闭环表现。','',f"Mixed Manifest SHA256：{out['manifest_sha256']}"]
 (EXP/'reports/03_mixed_critic_training.md').write_text('\n'.join(report)+'\n')
 print(json.dumps({'utc':out['utc'],'stage':out['stage'],'updates':{m:v['real_updates'] for m,v in critics.items()},'paired_prefix':n,'gate_passed':{m:bool(v['sameconfig_gate'] and v['sameconfig_gate']['passed']) for m,v in critics.items()},'actors':out['new_actor_updates'],'episodes':out['new_evaluation_episodes']}))
if __name__=='__main__':main()
