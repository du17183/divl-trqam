"""CPU-only coverage and explicitly defined small-perturbation ranking report."""
from pilot8_common_v1 import *
import numpy as np
import csv,shutil
from scipy.stats import spearmanr

def aggregate(rows,key):
    groups={t:[x for x in rows if x['task_id']==t and x.get(key) is not None] for t in range(10)}
    covered=[t for t,rr in groups.items() if rr]
    if not covered:return {'task_coverage':0,'available_task_mean':None,'all10_macro':None,'ci95':None}
    point=float(np.mean([np.mean([x[key] for x in groups[t]]) for t in covered]));samples=[];rng=np.random.default_rng(20261012)
    for _ in range(2000):
        values=[]
        for t in covered:
            rr=groups[t];sources=sorted({x['state_source_id'] for x in rr});chosen=rng.choice(sources,len(sources),True)
            values.append(np.mean([x[key] for source in chosen for x in rr if x['state_source_id']==source]))
        samples.append(np.mean(values))
    return {'task_coverage':len(covered),'available_task_mean':point,'all10_macro':point if len(covered)==10 else None,'ci95':list(map(float,np.quantile(samples,[.025,.975])))}

def main():
    assert read(EXP/'metrics/physical_action_audit002.json')['complete']
    rawpath=EXP/'metrics/physical_action_summary002.json';raw=read(rawpath);rows=[];tasks=[]
    for record in raw['records']:
        case=record['case'];rr=[x for x in record['result']['rows'] if x.get('valid') and x.get('ranking_candidate')]
        by={x['candidate']:x for x in rr};small=[by[x] for x in ['baseline','independent_small'] if x in by]
        for m in ['scalar','divl']:
            old=next(x for x in raw['states'] if x['case_id']==case['case_id'] and x['method']==m)
            distinct=len(small)==2 and abs(small[0]['return']-small[1]['return'])>1e-12
            near=int((small[0]['q'][m]-small[1]['q'][m])*(small[0]['return']-small[1]['return'])>0) if distinct else None
            rows.append({**old,'near_id':None,'near_id_definition':'baseline vs preregistered independent_small arm RMS0.03; exclude large/gripper candidates','near_id_ranking':near,'near_id_pairs':int(distinct),'near_id_controller7d_rms':by.get('independent_small',{}).get('effective_controller7d_rms'),'near_id_reason':None if distinct else 'Equal observed returns or missing branch'})
    for t in range(10):
        for m in ['scalar','divl']:
            rr=[x for x in rows if x['task_id']==t and x['method']==m]
            tasks.append({'task_id':t,'task_name':next(x['case']['task_name'] for x in raw['records'] if x['case']['task_id']==t),'method':m,'restored_states':sum(x['valid_candidates']==4 for x in rr),'physical_sources':len({x['state_source_id'] for x in rr}),'sortable_states':sum(x['ranking'] is not None for x in rr),'distinct_return_action_pairs':sum(x['action_pairs'] for x in rr),'pairwise_ranking_mean_per_state':float(np.mean([x['ranking'] for x in rr if x['ranking'] is not None])) if any(x['ranking'] is not None for x in rr) else None,'same_state_spearman':float(np.mean([x['spearman'] for x in rr if x['spearman'] is not None])) if any(x['spearman'] is not None for x in rr) else None,'near_id_sortable_states':sum(x['near_id_ranking'] is not None for x in rr),'near_id_ranking':float(np.mean([x['near_id_ranking'] for x in rr if x['near_id_ranking'] is not None])) if any(x['near_id_ranking'] is not None for x in rr) else None})
    allbranches=[x for r in raw['records'] for x in r['result']['rows'] if x.get('valid')]
    supplement={'created_utc':stamp(),'source_sha256':sha(rawpath),'source_file':str(rawpath),'planned_states':100,'restored_states':sum(r['result'].get('passed',False) for r in raw['records']),'physical_sources':len({x['state_source_id'] for x in rows}),'valid_branches':len(allbranches),'trace_hashes_audited':True,'prefix_state_maxabs':max(r['result']['recorded_prefix_state_max_absolute_error'] for r in raw['records']),'branch_state_maxabs':max(x['branch_state_error'] for x in allbranches),'hidden_controller_state_maxabs':max(x['hidden_state_error'] for x in allbranches),'per_task':tasks,'state_rows':rows,'summary':{m:{k:aggregate([x for x in rows if x['method']==m],k) for k in ['ranking','spearman','near_id_ranking']} for m in ['scalar','divl']},'near_id_scope':'Descriptive subset of already preregistered independent_small candidates. Reporting definition clarified after collection; not used for admission or tuning. Original generic near_id=True flag is not a measured distance gate.','unchanged_full_candidate_ranking_summary':raw['rank_summary'],'unchanged_physical_gradient_summary':raw['gradient_summary'],'diagnostic_data_not_added_to_training':True,'scientific_updates':0}
    for branch in allbranches:assert sha(branch['trace_path'])==branch['trace_sha256'],branch['trace_path']
    write(EXP/'metrics/physical_action_supplement003.json',supplement,True)
    target=EXP/'metrics/same_state_action_ranking.csv';archive=EXP/'provenance/resume_revision001/same_state_action_ranking_before_nearid_clarification.csv'
    assert not archive.exists();shutil.copy2(target,archive)
    with target.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=EXP/'reports/02_same_state_action_ranking.md';shutil.copy2(report,EXP/'provenance/resume_revision001/02_same_state_action_ranking_before_supplement.md')
    report.write_text('# 同状态独立候选动作排序\n\n100状态/60物理来源/550有效分支已完成。每任务10状态、来自6物理来源，不按100个独立场景处理。候选动作独立于Critic评分，所有同Return候选不制造胜负。\n\nNear-ID明确定义为原预注册baseline与independent_small（arm RMS0.03）候选，排除large及gripper；这是采集后的描述性口径澄清，未用于准入/调参。原泛化near_id=True标签不是实测距离准入，原结果完整保留。\n\n'+json.dumps({'summary':supplement['summary'],'per_task':tasks,'prefix_state_maxabs':supplement['prefix_state_maxabs'],'branch_state_maxabs':supplement['branch_state_maxabs'],'hidden_controller_state_maxabs':supplement['hidden_controller_state_maxabs']},ensure_ascii=False,indent=2)+'\n\n所有CI按任务分层、物理来源聚类。截断后缀为Observed Return，不是完整无限时域MC。反事实数据不加入训练。\n')
    write(EXP/'provenance/physical_cpu_supplement_sources009.json',{'created_utc':stamp(),'files':{Path(__file__).relative_to(ROOT).as_posix():sha(__file__),rawpath.relative_to(ROOT).as_posix():sha(rawpath)},'outputs':{p.relative_to(ROOT).as_posix():sha(p) for p in [EXP/'metrics/physical_action_supplement003.json',target,report]},'no_actor_or_critic_or_replay_changes':True},True)
    print(json.dumps({k:supplement[k] for k in ['restored_states','physical_sources','valid_branches','prefix_state_maxabs','branch_state_maxabs','hidden_controller_state_maxabs','summary']}))
if __name__=='__main__':main()
