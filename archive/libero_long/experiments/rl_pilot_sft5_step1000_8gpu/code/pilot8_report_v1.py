"""Actual episode/checkpoint/log-derived pilot tables; unavailable is never zero."""
from pilot8_common_v1 import *
import csv
import numpy as np

METHODS={'A':'Frozen SFT1000','B':'BC','B2':'BC EMA','C':'Scalar QAM','D':'DIVL QAM','E':'Scalar BC QAM','F':'DIVL BC QAM','G':'Scalar TRQAM','H':'DIVL TRQAM'}


def csvfile(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:('N/A' if v is None else v) for k,v in r.items()})


def bootstrap(a,b=None,seed=20261010):
    """Fixed ten-task equal weights, resample physical initial sources per task."""
    rng=np.random.default_rng(seed);values=[]
    for t in range(10):
        rr=a[t];other={r['state_source_id']:r for r in b[t]} if b else None
        groups={}
        for r in rr:
            v=float(r['success'])-(float(other[r['state_source_id']]['success']) if other else 0.)
            groups.setdefault(r['state_source_id'],[]).append(v)
        x=np.array([np.mean(v) for v in groups.values()]);assert len(x)>0
        values.append(np.mean(x[rng.integers(len(x),size=(10000,len(x)))],axis=1))
    return [float(x) for x in np.quantile(np.mean(values,axis=0),[.025,.975])]


def task_ci(rows,seed):
    groups={}
    for r in rows:groups.setdefault(r['state_source_id'],[]).append(float(r['success']))
    x=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed)
    return [float(v) for v in np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1),[.025,.975])]


def actual_evaluation(method):
    result={t:[] for t in range(10)}
    evaluation_id='A_frozen_repair_standard50' if method=='A' else f'{method}_step000100_standard50'
    for p in (EXP/'evaluation'/evaluation_id).glob('task_*_shard_*.json'):
        d=read(p)
        if d['complete']:
            assert sha(d['trace_hdf5'])==d['trace_sha256'];result[d['task_id']].extend(d['records'])
    if not all(len(v)==50 and len({x['official_index'] for x in v})==50 for v in result.values()):return None
    proto=read(EXP/'configs'/f'evaluation_{method}_100.json');base=read(EXP/'configs/evaluation_protocol_original001.json')
    for t in range(10):
        expected={r['official_index']:r for r in base['tasks'][t]['episodes']}
        for r in result[t]:
            for key in ['state_source_id','initial_vector_sha256','environment_seed']:assert r[key]==expected[r['official_index']][key]
            assert r['policy_seed']==20261009 and r['inference_checkpoint_manifest_sha256']==proto['checkpoint_manifest_sha256']
    return result


def main():
    split=read(ROOT/'configs/libero_10_split_manifest.json');names=[x['task_name'] for x in split['tasks']]
    stage=read(EXP/'metrics/pipeline_stage008.json') if (EXP/'metrics/pipeline_stage008.json').exists() else {'stage':'WAIT_STRICT_REPAIR_GATE','complete':False}
    repair=read(EXP/'metrics/repair_stage001.json') if (EXP/'metrics/repair_stage001.json').exists() else None
    stage['repair_gate_status']=repair
    stage['bc_data_repair_gate']=read(EXP/'metrics/bc_data_repair_gate001.json')
    evals={m:actual_evaluation(m) for m in METHODS};baseline=evals['A'];success=[];returns=[];lengths=[];methodrows=[]
    for m in METHODS:
        by=evals[m];summary=None
        if by:
            summaries=[]
            for t,rr in by.items():
                old={r['state_source_id']:r for r in baseline[t]};summary={
                    'method':m,'task_id':t,'task_name':names[t],'episodes':50,'successes':sum(r['success'] for r in rr),
                    'success_rate':float(np.mean([r['success'] for r in rr])),'success_ci95':task_ci(rr,20261010+t),'mean_return':float(np.mean([r['return'] for r in rr])),
                    'mean_episode_length':float(np.mean([r['episode_length'] for r in rr])),
                    'frozen_success_to_failure':sum(old[r['state_source_id']]['success'] and not r['success'] for r in rr),
                    'frozen_failure_to_success':sum(not old[r['state_source_id']]['success'] and r['success'] for r in rr)}
                summaries.append(summary)
            summary={'method':m,'name':METHODS[m],'status':'ACTUALLY_EVALUATED','success_rate':float(np.mean([x['success_rate'] for x in summaries])),
                'successes':sum(x['successes'] for x in summaries),'episodes':500,'ci95':bootstrap(by),
                'mean_return':float(np.mean([x['mean_return'] for x in summaries])),'mean_episode_length':float(np.mean([x['mean_episode_length'] for x in summaries])),
                'per_task':summaries,'frozen_success_to_failure':sum(x['frozen_success_to_failure'] for x in summaries),'frozen_failure_to_success':sum(x['frozen_failure_to_success'] for x in summaries)}
        else:summary={'method':m,'name':METHODS[m],'status':stage.get('methods',{}).get(m,{}).get('status','PENDING'),'success_rate':None,'episodes':0,'ci95':None,'mean_return':None,'mean_episode_length':None,'per_task':[]}
        train=EXP/'metrics/actor'/f"{'B' if m=='B2' else m}_train001/result.json"
        if train.exists():
            v=read(train);summary.update(actor_updates=v['scientific_actor_updates'],actor_global_batch=v['batch'],
                actor_train_allocated_gpu_hours=0 if m=='B2' else v['allocated_gpu_hours'],ema_shared_B_cost=m=='B2',
                actor_train_wall_s=v['wall_s'],drift=v.get('drift'),seconds_per_update=v.get('steady_total_s'))
        elif m=='A':summary.update(actor_updates=0,actor_train_allocated_gpu_hours=0)
        methodrows.append(summary)
    for t in range(10):
        sr=rr=ll=None;sr={'task_id':t,'task_name':names[t]};rr=sr.copy();ll=sr.copy()
        for m in METHODS:
            data=next(x for x in methodrows if x['method']==m)['per_task'];v=data[t] if data else None
            sr[m]=f"{v['success_rate']*100:.1f}% ({v['successes']}/50)" if v else None
            rr[m]=v['mean_return'] if v else None;ll[m]=v['mean_episode_length'] if v else None
        success.append(sr);returns.append(rr);lengths.append(ll)
    success.append({'task_name':'10-task macro mean',**{x['method']:(f"{x['success_rate']*100:.1f}% ({x['successes']}/500)" if x['success_rate'] is not None else None) for x in methodrows}})
    returns.append({'task_name':'10-task macro mean',**{x['method']:x['mean_return'] for x in methodrows}});lengths.append({'task_name':'10-task macro mean',**{x['method']:x['mean_episode_length'] for x in methodrows}})
    paired=[]
    for x,y in [('B','A'),('B2','A'),('C','A'),('D','A'),('D','C'),('E','C'),('F','D'),('G','C'),('H','D'),('H','G'),('C','B'),('D','B')]:
        a,b=evals[x],evals[y];delta=None;ci=None
        if a and b:
            delta=float(np.mean([np.mean([r['success'] for r in a[t]])-np.mean([r['success'] for r in b[t]]) for t in range(10)]))*100;ci=[z*100 for z in bootstrap(a,b)]
        paired.append({'comparison':x+'-'+y,'difference_pp':delta,'ci95_low_pp':ci[0] if ci else None,'ci95_high_pp':ci[1] if ci else None,'status':'ACTUALLY_PAIRED' if ci else 'N/A'})
    for fn,rows in [('eight_methods_per_task_success.csv',success),('eight_methods_sft1000_8gpu_success.csv',success),('eight_methods_per_task_return.csv',returns),('eight_methods_per_task_length.csv',lengths)]:csvfile(EXP/'metrics'/fn,rows,['task_id','task_name',*METHODS])
    csvfile(EXP/'metrics/eight_methods_paired_comparison.csv',paired,['comparison','difference_pp','ci95_low_pp','ci95_high_pp','status'])
    critics=[];actorlog=[];kl=[];resource=[];replay=[]
    for method in ['scalar','divl']:
        p=EXP/'metrics/critic'/f'{method}_diagnose001/result.json'
        if p.exists():critics.extend({'method':method,**x} for x in read(p).get('per_task',[]))
        else:critics.extend({'method':method,'task_id':t,'task_name':names[t],'pairwise_action_ranking':None,'near_id_action_ranking':None,'valid_action_pairs':0,'positive_gradient_vs_random_true_return':None} for t in range(10))
    if (EXP/'metrics/replay_quality001.json').exists():replay=read(EXP/'metrics/replay_quality001.json')['by_task']
    for p in (EXP/'metrics/actor').glob('*_train001/train_steps.jsonl'):
        for line in p.read_text().splitlines():
            d=json.loads(line);actorlog.append({k:v for k,v in d.items() if not isinstance(v,(dict,list))})
            if 'path_kl' in d:kl.append({k:d.get(k) for k in ['method','update','path_kl','path_kl_ema','lambda_effective','kl_budget','kl_constraint_violation']})
    for d in list((EXP/'metrics/critic').glob('*/result.json'))+list((EXP/'metrics/actor').glob('*/result.json')):
        resource.extend({'run':d.parent.name,**x} for x in read(d).get('telemetry',[]))
    fields=sorted({k for x in critics for k in x});csvfile(EXP/'metrics/critic_per_task_metrics.csv',critics,fields)
    csvfile(EXP/'metrics/replay_distribution.csv',replay,['task_id','task_name','episodes','successes','censored'])
    csvfile(EXP/'metrics/actor_training_metrics.csv',actorlog,sorted({k for x in actorlog for k in x}) or ['method','update'])
    csvfile(EXP/'metrics/trqam_kl_metrics.csv',kl,['method','update','path_kl','path_kl_ema','lambda_effective','kl_budget','kl_constraint_violation'])
    csvfile(EXP/'metrics/gpu_resource_usage.csv',resource,['run','utc','gpu','memory_mib','util_percent','power_w'])
    result={'asof_utc':stamp(),'stage':stage['stage'],'complete':stage.get('complete',False),'methods':methodrows,'paired':paired,
        'critic_per_task':critics,'replay_distribution':replay,'scientific_limits':['One training seed, exploratory','Official50 reused development benchmark; not untouched Final',
        'Ranking/near-ID/physical-gradient N/A without controlled same-state counterfactual pairs','Censored return observed suffix, full MC unknown'],
        'provenance':{'checkpoint_manifest_sha256':config()['sft_checkpoint_manifest_sha256'],'normalization_sha256':sha(config()['normalization']),
                      'protocol_sha256':sha(EXP/'configs/protocol001.json'),'source_request_sha256':sha(EXP/'provenance/user_request.txt')},
        'pipeline':stage,'source':'actual checkpoints, scientific update logs and complete native episode JSON/HDF5 files',
        'training_cost_scope':'actual worker allocated GPU-hours include read/compute/save/hash; inference+SIM shared-card costs must not be summed as independent GPU allocations',
        'trqam_scope':'official discrete shifted grid/path KL/dual/slow BC EMA port; native VLA FM, minDoubleQ and fixed pretrained IQL critic are declared adaptations; logged KL is pre-update finite-step estimate, not guaranteed exact continuous or post-update KL'}
    write(EXP/'metrics/all_metrics.json',result)
    text='# GPU 0–7 八组 Pilot 实测进度\n\n更新时间：'+result['asof_utc']+'。当前：`'+result['stage']+'`。一训练 Seed，探索性结果。\n\n|Task|'+ '|'.join(METHODS)+'|\n|---|'+ '|'.join(['---']*len(METHODS))+'|\n'
    for row in success:text+='|'+row['task_name']+'|'+'|'.join(str(row.get(m) or 'N/A') for m in METHODS)+'|\n'
    text+='\n原生 official50 为重复开发用标准基准，不是严格未见 Final。只有完整500 Episode才列算法闭环数值，未运行不填0。配对95%CI按任务等权分层、独立物理初始来源聚类。\n'
    text+='\n历史 Frozen A500 为211/500（42.2%），原记录已归档。由于本次编译运行配置改变，本轮复用重新评测后的208/500（41.6%）原始记录；新8卡推理路由还须通过固定输入动作一致性检查。\n'
    (EXP/'reports/eight_methods_per_task.md').write_text(text)
    (EXP/'reports/current_stage_summary.md').write_text(text)
    if stage.get('complete'):
        reasons=json.dumps(stage.get('methods',{}),ensure_ascii=False,indent=2)
        lookup={x['method']:x for x in methodrows};rate=lambda m: f"{lookup[m]['success_rate']*100:.1f}% ({lookup[m]['successes']}/500)" if lookup[m]['success_rate'] is not None else 'N/A，详见支路停止／未完成状态'
        actor=stage.get('actor_execution',{});summary=''
        answers=[
            '完整π0.5反向传播是否通过：'+str(actor.get('passed',False))+'；以各方法完整FP32参数/Adam/EMA/恢复JIT Gate为准，失败支路不算通过。',
            '准入Global Batch：'+str(actor.get('batch','N/A'))+'，实际设备数：'+str(actor.get('devices','N/A'))+'；实际吞吐候选见configs/actor_execution001.json。',
            'Replay成败统计：'+str(replay)+'。全部原始300轨迹保留；240 Train/60诊断Held-out源已预先冻结。',
            'Scalar动作排序：N/A。没有经过完整状态／控制器历史恢复验收的同状态动作Pair。',
            'DIVL动作排序：N/A，理由同上。',
            '更有效的真实动作梯度：N/A；Critic相关性与合成Q梯度不替代真实环境Return。',
            'BC-only：'+rate('B')+'；B2 EMA：'+rate('B2')+'。',
            'Scalar＋QAM：'+rate('C')+'。',
            'DIVL＋QAM：'+rate('D')+'。',
            'DIVL相对Scalar：查看D-C的配对CI；仅单Seed工程Pilot，不能宣称跨训练Seed可靠收益。',
            '显式BC约束：查看E-C、F-D和Frozen成功→失败数量；未执行对照为N/A。',
            'TRQAM路径KL：以原始未裁剪path_kl及预算超标比例为准；对偶估计裁剪不能隐藏超标。有限100步不保证严格预算满足。',
            'TRQAM闭环收益：查看G-C、H-D；未执行或CI跨0则没有充分证据。',
            '全部逐任务成功率：见本报告完整Task Name表及eight_methods_per_task_success.csv。',
            '宏平均：'+str({m:rate(m) for m in METHODS})+'。',
            '任务改善／退化：以每任务差值及原成功→失败、失败→成功配对计数为准；方法未评测不推测。',
            '各组实际训练GPU小时：'+str({m:lookup[m].get('actor_train_allocated_gpu_hours') for m in METHODS})+'；候选Profiler/Gate单列，不算科学更新。',
            '瓶颈：由实际steady_read_s/steady_compute_s、GPU遥测、采集RPC/环境步耗时判定，不用显存占用假称算力利用。',
            '是否扩大多Seed：当前物理动作排序／梯度缺失，不自动扩大。正闭环差值可作后续候选，需独立验证。',
            '在线LWD：当前只有冻结部署Replay上的Critic预训练与最多100步Actor工程Pilot，没有依据直接进入在线LWD。']
        summary='\n## 二十问逐项回答\n\n'+'\n\n'.join(f'{i+1}. {a}' for i,a in enumerate(answers))
        (EXP/'reports/pilot_final_conclusion.md').write_text(text+'\n## 支路完成／停止依据\n\n```json\n'+reasons+'\n```\n'+summary+'\n')
    print(json.dumps({'stage':result['stage'],'complete':result['complete'],'methods_actual': [x['method'] for x in methodrows if x['episodes']==500]}))


if __name__=='__main__':main()
