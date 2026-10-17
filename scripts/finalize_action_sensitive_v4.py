"""Thirty-four requested answers with nulls for unexecuted formal stages."""
import json,pathlib,datetime,shutil
from action_sensitive_v4_common import R,D,O,REP,atomic,sha
def load(p):
 return json.loads(p.read_text()) if p.exists() else None
def est(x,percent=False):
 if not x or x.get('mean') is None:return '未有可用估计'
 m=100 if percent else 1;u='%' if percent else ''
 return f'{x["mean"]*m:.4f}{u}，95%CI [{x["ci95"][0]*m:.4f},{x["ci95"][1]*m:.4f}]{u}'
def main():
 REP.mkdir(parents=True,exist_ok=True)
 offline=load(O/'offline_complete.json');audit=load(O/'supervision.json');probe=load(R/'artifacts/v4_representation_probe.json')
 status=load(O/'analysis_status.json') or load(O/'collection_status.json') or {};manifest=load(D/'manifest.json')
 answers={i:'尚未完成；不使用历史结果冒充本轮结果。' for i in range(1,35)}
 if manifest:answers[1]=str(manifest['states']);answers[2]=str({str(t):len({r['state_id'] for r in manifest['records'] if r['task_id']==t}) for t in range(10)});answers[3]=str(manifest['outcomes'])
 if audit:
  s=audit['summary'];types=audit['candidate_types']
  for i,b in [(4,'Near-ID'),(5,'Mid'),(6,'Far-OOD')]:answers[i]=f'{s["support_counts"][b]}/{s["outcomes"]} = {s["support_counts"][b]/s["outcomes"]:.2%}'
  answers[7]=str(s['near_effective_states'])+'；定义：该state至少一组within-Near-ID pair |deltaG|>1e-6。'
  answers[8]=str(s['near_actions_per_state']);answers[9]=json.dumps(s['near_supervision'])
  for i,t in [(10,'0.0001'),(11,'0.001'),(12,'0.01')]:answers[i]=json.dumps(s['near_supervision'][t])
  answers[13]=f'{s["mixed_success_failure_states"]}/{s["states"]} = {s["mixed_success_failure_states"]/s["states"]:.2%}'
  for i,k in [(14,'interpolation'),(15,'local_noise'),(16,'local_pca')]:answers[i]=json.dumps(types[k])
  answers[17]='候选机制全部保留。比较Near-ID的连续deltaG，不按收益挑动作/删数据：'+json.dumps({k:v['near_different_from_behavior'] for k,v in types.items()})
 if probe:
  probes=probe['probes']
  for i,name in [(18,'A'),(19,'B'),(20,'C'),(21,'D')]:answers[i]='Overall '+est(probes[name]['metrics']['all']['rank'],True)+'；Near-ID '+est(probes[name]['metrics']['Near-ID']['rank'],True)+'；Spearman '+est(probes[name]['metrics']['all']['spearman'])
  best=max(probes,key=lambda n:probes[n]['metrics']['all']['rank']['mean'] or -1)
  answers[22]=f'描述性最好为{best}；不以最好probe事后选择primary。预注册primary始终为B，GT D不得进入正式算法。'
  answers[23]=probe['decision']+'；GT−B paired '+est(probe['paired']['D_minus_B']['all']['rank'],True)+'。Near-ID覆盖属于经验代理，不能直接证明物理OOD；固定GT/PCA诊断不构成理论上限。'
  b=probes['B']['metrics'];answers[24]=str(b['all']['rank']['mean']>=.60);answers[25]=str((b['Near-ID']['rank']['mean'] or 0)>=.65)
  answers[26]=str(b['Near-ID']['rank']['states'])+'；是否>=100：'+str(b['Near-ID']['rank']['states']>=100)
  answers[27]=str(probe['gate_passed'])+'；规则：'+probe['protocol']['gate']
 evals={mix:load(O/f'v4_counterfactual_{mix}.json') for mix in ['R1','R2']}
 grads={mix:load(O/f'v4_new_gradient_{mix}.json') for mix in ['R1','R2']}
 for i,m in [(28,'scalar'),(29,'divl')]:
  answers[i]='\n\n'.join(mix+': '+(est(e['all_budgets'][m]['2000']['all']['rank'],True) if e else '未正式评估/未训练') for mix,e in evals.items())
 answers[30]='\n\n'.join(mix+': '+(est(e['paired_divl_minus_scalar']['2000']['all']['rank'],True)+'（差值单位百分点）' if e else '未有v4正式比较') for mix,e in evals.items())
 answers[31]='\n\n'.join(mix+': '+(', '.join(m+'='+est(e['all_budgets'][m]['2000']['Near-ID']['rank'],True) for m in ['scalar','divl']) if e else '未正式评估') for mix,e in evals.items())
 answers[32]='\n\n'.join(mix+': '+json.dumps(g['methods']) for mix,g in grads.items() if g) or '未模拟v4新critic方向；旧600方向结果不能移作v4证据。'
 permitted=[(mix,m) for mix,g in grads.items() if g for m,v in g['methods'].items() if v['passed']]
 answers[33]=str(bool(permitted))+'；本轮无论何种结果都不启动Actor/QAM。'
 decision=status.get('stage',probe['decision'] if probe else 'V4_RUNNING')
 answers[34]=decision
 titles=['v4 states','每task states','总branch数','Near-ID占比','Mid占比','Far-OOD占比','Near-ID有效states','每state Near-ID动作数','Near-ID可排序pairs','Near |deltaG|>1e-4','Near |deltaG|>1e-3','Near |deltaG|>1e-2','混合success/failure states','插值效果','局部noise效果','局部PCA效果','ID范围有效future差异','Probe A','Probe B','Probe C','Probe D GT','最好representation','瓶颈判断','Overall>=60%','Near-ID>=65%','Near-ID有效held-out>=100','是否允许正式v4 critic','Scalar2000ranking','DIVL2000ranking','pairedDIVL-ScalarCI','两组Near-IDranking','真实gradient广泛性','fixed-referenceQAM资格','最终状态']
 text='# Action-sensitive replay v4：当前完整进度\n\n'+datetime.datetime.now().astimezone().isoformat()+'；b300-1 `'+str(R)+'`\n\n**'+decision+'**；Actor/QAM/dynamic beta冻结。\n\n'
 for i,title in enumerate(titles,1):text+=f'## {i}. {title}\n\n{answers[i]}\n\n'
 text+='## 固定协议与解释边界\n\n实际冻结π0.5图像权重用于A/B及局部近邻生成；Near-ID仍使用旧SigLIP公共邻域与原p50/p90/p99阈值。故Near-ID不是严格物理OOD结论。所有300个旧预选train anchors复用，不看v3/v4分支结果选状态。动作7D、H10执行5、padding25维零、γ.9999、native reward/controller/checkpoint保持。五fold按完整(task,initialID)分组，无dev/final训练。GT只为诊断，正式replay loader不读取GT。\n'
 if manifest:text+='\nManifest SHA256：`'+sha(D/'manifest.json')+'`；config SHA256：`'+sha(D/'protocol.json')+'`。\n'
 (REP/'current_stage_summary.md').write_text(text)
 history=O/'historical_current_stage_summary_v3.md'
 if not history.exists() and (R/'reports/current_stage_summary.md').exists():shutil.copy2(R/'reports/current_stage_summary.md',history)
 (R/'reports/current_stage_summary.md').write_text(text)
 for name in ['v4_candidate_generation.md','v4_near_id_coverage.md','v4_branch_collection.md','v4_action_supervision.md','v4_representation_probe.md','v4_positive_control.md']:
  if (REP/name).exists():shutil.copy2(REP/name,R/'reports'/name)
 for method in ['scalar','divl']:
  parts=[]
  for mix,e in evals.items():
   p=R/'reports'/f'{method}_v4_{mix}_critic_report.md'
   if p.exists():parts.append(p.read_text())
  (REP/f'{method}_v4_critic.md').write_text('# '+method+' v4 critic\n\n'+('\n\n'.join(parts) if parts else '未训练/未正式评估。本轮正对照门槛在正式训练之前执行；没有用v2/v3替代v4结果。'))
 if any(evals.values()):
  text='\n\n'.join((R/'reports'/f'v4_heldout_{mix}.md').read_text() for mix,e in evals.items() if e)
  prior=load(R/'artifacts/replay_action_audit_v3/v2_counterfactual_evaluation.json')
  text+='\n\n## V2 to V4 changes\n\n'
  if prior:
   for mix,e in evals.items():
    if e:text+=mix+': '+', '.join(m+'='+f'{100*(e["all_budgets"][m]["2000"]["all"]["rank"]["mean"]-prior["all_budgets"][m]["2000"]["all"]["rank"]["mean"]):+.3f}pp' for m in ['scalar','divl'])+'\n'
  (REP/'v4_heldout_evaluation.md').write_text(text)
 if any(grads.values()): (REP/'v4_gradient_direction.md').write_text(json.dumps(grads,indent=2))
 atomic(R/'artifacts/v4_critic_metrics.json',dict(stage=decision,mixes=evals,gradients=grads,actor_updates=False))
 print('V4_REPORTS_UPDATED',decision,flush=True)
if __name__=='__main__':main()
