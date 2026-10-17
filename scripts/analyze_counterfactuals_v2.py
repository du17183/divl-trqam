# Public release: deployment paths anonymized; algorithm equations unchanged.
"""State-clustered paired action evidence from the complete version2 protocol."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path('/opt/divl-trqam')
EPS=[.01,.03,.05]
TIE=1e-6

def safe(v):
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [safe(x) for x in v]
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    return v

def dump(v):return json.dumps(safe(v),indent=2,allow_nan=False)
def describe(x,draws):
    x=np.asarray(x,float);x=x[np.isfinite(x)];draws=np.asarray(draws,float);draws=draws[np.isfinite(draws)]
    return {'mean':float(x.mean()) if len(x) else None,'median':float(np.median(x)) if len(x) else None,
            'n_states':len(x),'ci95':np.quantile(draws,[.025,.975]).tolist() if len(draws) else [None,None],
            'distribution':x.tolist()}

def main():
    assert json.loads((ROOT/'artifacts/reproduction_audit.json').read_text())['passed']
    align=json.loads((ROOT/'artifacts/q_alignment_audit.json').read_text());assert align['alignment_passed']
    groups=json.loads((ROOT/'artifacts/counterfactual_results_v2.json').read_text())
    assert len(groups)==50 and len({g['metadata']['state_id'] for g in groups})==50
    groups.sort(key=lambda g:g['metadata']['state_id'])
    for g in groups:
        assert g['status']=='valid' and g['snapshot_version']==2 and g['reproduction']['reproduction_pass']
        assert len(g['branches'])==20
    flattened=[{**b,'snapshot_version':2,'reproduction_pass':True} for g in groups for b in g['branches']]
    assert sum(b['kind']=='candidate' for b in flattened)==400
    assert sum(b['kind']=='gradient' for b in flattened)==600
    pd.DataFrame(flattened).to_parquet(ROOT/'artifacts/counterfactual_final.parquet',index=False)
    task=np.asarray([g['metadata']['task_id'] for g in groups]);rng=np.random.default_rng(20261007)
    bootstrap=np.asarray([np.concatenate([rng.choice(np.flatnonzero(task==t),5,replace=True) for t in range(10)]) for _ in range(5000)])
    method_results={};all_budgets={};vectors={};grad_results={}
    for method in ['scalar','divl']:
        budget=[]
        for step in [250,500,1000,2000]:
            static=json.loads((ROOT/f'artifacts/static_action_{method}_step{step}.json').read_text())
            lookup={r['state_id']:r for r in static['records']};states=[]
            for group in groups:
                cand=sorted((r for r in group['branches'] if r['kind']=='candidate'),key=lambda r:r['candidate_index'])
                record=lookup[group['metadata']['state_id']]
                with np.load(record['path'],allow_pickle=False) as f:
                    q=np.asarray(f['candidate_q']);np.testing.assert_array_equal(f['candidate_actions'],np.asarray([r['action'] for r in cand],np.float32))
                g=np.asarray([r['future_return'] for r in cand]);count=0;correct=0.
                for a,b in itertools.combinations(range(8),2):
                    if abs(g[a]-g[b])<=TIE:continue
                    count+=1;correct+=.5 if abs(q[a]-q[b])<=1e-10 else float((q[a]-q[b])*(g[a]-g[b])>0)
                rho=float(spearmanr(q,g).statistic) if g.std()>TIE and q.std()>1e-10 else np.nan
                states.append({'state_id':group['metadata']['state_id'],'task':group['metadata']['task_id'],
                               'ranking':correct/count if count else np.nan,'effective_pairs':count,'correct_pairs':correct,
                               'spearman':rho,'future_variance':float(g.var()),'returns':g.tolist(),'q':q.tolist()})
            ranks=np.asarray([v['ranking'] for v in states]);rhos=np.asarray([v['spearman'] for v in states])
            counts=np.asarray([v['effective_pairs'] for v in states]);correct=np.asarray([v['correct_pairs'] for v in states])
            rankdraw=np.nanmean(ranks[bootstrap],axis=1);rhodraw=np.nanmean(rhos[bootstrap],axis=1)
            pooleddraw=np.sum(correct[bootstrap],axis=1)/np.maximum(np.sum(counts[bootstrap],axis=1),1)
            result={'steps':step,'state_macro_ranking':describe(ranks,rankdraw),'state_wise_spearman':describe(rhos,rhodraw),
                    'pair_weighted_ranking':float(correct.sum()/counts.sum()) if counts.sum() else None,
                    'pair_weighted_ci95':np.quantile(pooleddraw,[.025,.975]).tolist(),'effective_pairs':int(counts.sum()),
                    'ties_excluded':1400-int(counts.sum()),'static_rho':static['rho'],
                    'shuffle_mean_absolute_delta':static['shuffle_mean_absolute_delta'],'gradient_norm_mean':static['gradient_norm_mean'],
                    'shuffle_relative_q_std':static['shuffle_mean_absolute_delta']/max(np.sqrt(static['total_q_variance']),1e-12),
                    'per_state':states}
            budget.append(result)
            if step==2000:
                vectors[method]={'ranking':ranks,'spearman':rhos,'ranking_draw':rankdraw,'spearman_draw':rhodraw}
                method_results[method]=result
        all_budgets[method]=budget
        eps_metrics={};delta_matrix=[]
        for eps in EPS:
            delta=[]
            for group in groups:
                pair=[b for b in group['branches'] if b['kind']=='gradient' and b['method']==method and b['epsilon']==eps]
                assert len(pair)==2
                delta.append(next(b['future_return'] for b in pair if b['direction']==1)-next(b['future_return'] for b in pair if b['direction']==-1))
            delta=np.asarray(delta);delta_matrix.append(delta)
            eps_metrics[str(eps)]={**describe(delta,delta[bootstrap].mean(axis=1)),
                                   'positive_ratio':float((delta>0).mean()),'zero_ratio':float((delta==0).mean()),
                                   'positive_ratio_ci95':np.quantile((delta[bootstrap]>0).mean(axis=1),[.025,.975]).tolist()}
        delta=np.mean(delta_matrix,axis=0);vectors[method]['gradient']=delta
        vectors[method]['gradient_draw']=delta[bootstrap].mean(axis=1)
        grad_results[method]={'by_epsilon':eps_metrics,'across_epsilon_state_mean':{**describe(delta,delta[bootstrap].mean(axis=1)),
                                                                                 'positive_ratio':float((delta>0).mean())},
                              'unit':'native discounted future return, G(a+) minus G(a-); whole states resampled, eps not independent samples'}
    paired={}
    for name in ['ranking','spearman','gradient']:
        delta=vectors['divl'][name]-vectors['scalar'][name]
        draws=np.nanmean(delta[bootstrap],axis=1)
        paired[name+'_divl_minus_scalar']=describe(delta,draws)
        paired[name+'_divl_minus_scalar']['median_ci95']=np.nanquantile(np.nanmedian(delta[bootstrap],axis=1),[.025,.975]).tolist()
    gates={}
    for method,result in method_results.items():
        grad=grad_results[method]['across_epsilon_state_mean'];rank=result['state_macro_ranking'];rho=result['state_wise_spearman']
        coverage=rank['n_states']>=25 and rho['n_states']>=25 and result['effective_pairs']>=200
        gates[method]={'coverage_passed':coverage,
                       'ranking_positive_evidence':rank['mean'] is not None and rank['mean']>=.60 and rank['ci95'][0]>.5,
                       'spearman_positive_evidence':rho['ci95'][0] is not None and rho['ci95'][0]>0,
                       'real_gradient_positive_evidence':grad['ci95'][0]>0,
                       'meaningful_action_shuffle':result['shuffle_relative_q_std']>.01}
        # Show robustness to any one state, in addition to state-bootstrap.
        arr=vectors[method]['gradient'];gates[method]['leave_one_state_out_gradient_mean_range']=[float(min(np.delete(arr,i).mean() for i in range(50))),float(max(np.delete(arr,i).mean() for i in range(50)))]
        top=np.argsort(np.abs(arr))[::-1][:2]
        positives=np.sort(np.maximum(arr,0))[::-1]
        grad_results[method]['influence_diagnostic']={
          'largest_two_absolute_state_effects':[{'state_id':groups[int(i)]['metadata']['state_id'],'mean_delta_g':float(arr[i])} for i in top],
          'share_of_all_positive_delta_from_top2':float(positives[:2].sum()/max(positives.sum(),1e-15)),
          'mean_after_omitting_top2_absolute_effects':float(np.delete(arr,top).mean()),
          'note':'Post hoc influence diagnostic only; all50 states remain in primary estimates and bootstrap. No branch or difficult state was removed.'}
        gates[method]['gradient_survives_removing_any_one_state']=gates[method]['leave_one_state_out_gradient_mean_range'][0]>0
        gates[method]['passed']=all(v for k,v in gates[method].items() if k!='leave_one_state_out_gradient_mean_range')
    decision='EVIDENCE_FOR_NEXT_STAGE' if any(d['passed'] for d in gates.values()) else 'STOP_ACTOR_QAM'
    ranking={'methods':method_results,'paired_comparison':paired,'all_budgets':all_budgets,'bootstrap':'5000 paired whole-state resamples, task-stratified five states/task; not action pairs',
             'states':50,'candidate_branches':400,'tie_tolerance':TIE,'engineering_gate':gates,'decision':decision,
             'note':'Engineering60% reference is not a paper significance threshold. Final2000 budget fixed before results; no best-checkpoint selection.'}
    ranking['candidate_protocol']='Unchanged behavior, frozen actor fresh noise, nearest train replay by proprio, normalized random ±.03/±.10, gripper flip. Existing frozen arrays reused; no candidate regeneration.'
    truth=[sorted((b for b in group['branches'] if b['kind']=='candidate'),key=lambda b:b['candidate_index']) for group in groups]
    ranking['future_structure']={'states_with_success_failure_candidate_mixture':sum(any(b['success'] for b in rows) and any(not b['success'] for b in rows) for rows in truth),
        'all_success_candidate_states':sum(all(b['success'] for b in rows) for rows in truth),
        'all_failure_candidate_states':sum(all(not b['success'] for b in rows) for rows in truth),
        'candidate_success_ratio':float(np.mean([b['success'] for rows in truth for b in rows])),
        'interpretation':'Native sparse reward and first-success stopping: positive return differences mostly encode success latency;0 versuspositive encodes success/failure. One frozen continuation-noise stream peraction; not a repeated-noise estimate of P(G|s,a).'}
    gradient={'methods':grad_results,'paired_comparison':paired['gradient_divl_minus_scalar'],'states':50,'gradient_branches':600,'epsilon':EPS,'decision':decision,
      'epsilon_space':'Unit-norm gradient in original quantile-normalized critic action space; map through unchanged official denormalization and native[-1,1] clipping. Epsilon is not a physical distance/angle.',
      'endpoint_semantics':'Critic and real branches evaluate first k executed steps (k<=5) with unchanged10-step array/mask; continuation policy/noise is shared.'}
    (ROOT/'artifacts/action_ranking_metrics.json').write_text(dump(ranking))
    (ROOT/'artifacts/gradient_direction_metrics.json').write_text(dump(gradient))
    (ROOT/'artifacts/critic_action_metrics_v2.json').write_text(dump({'ranking':ranking,'gradient':gradient,'decision':decision}))
    text='# Same-state action ranking\n\n| Method | State mean ranking | Ranking95%CI | Mean state Spearman | Spearman95%CI |\n|---|---:|---|---:|---|\n'
    for method,d in method_results.items():
        text+=f"| {method} | {d['state_macro_ranking']['mean']} | {d['state_macro_ranking']['ci95']} | {d['state_wise_spearman']['mean']} | {d['state_wise_spearman']['ci95']} |\n"
    text+='\nPaired DIVL-minus-Scalar results:\n\n'+dump(paired)+'\n\nAll retained budgets:\n\n'
    for method,budgets in all_budgets.items():
        text+='\n'+method+'\n\n'+dump([{k:v for k,v in d.items() if k!='per_state'} for d in budgets])+'\n'
    text+='\nDecision: '+decision+'\n\n'+ranking['bootstrap']+'\n'
    text+='\nActual candidate future structure:\n\n'+dump(ranking['future_structure'])+'\n'
    (ROOT/'reports/critic_action_ranking_report.md').write_text(text)
    (ROOT/'reports/qam_gradient_direction_report.md').write_text('# Real QAM gradient direction\n\n'+dump(gradient)+'\n\nNo actor update was performed. This uses true environment return, not predictedQ increases.\n')
    (ROOT/'reports/counterfactual_final_report.md').write_text('# Final version2 counterfactual set\n\n50 fixed states,400 candidate branches,600 gradient branches. All per-state reproduction gates passed. All old35 states rerun; original700 provisional branches retained.\n\n'+text)
    print('ACTION_EVIDENCE_COMPLETE',dump({'methods':{m:{k:v for k,v in d.items() if k!='per_state'} for m,d in method_results.items()},'gates':gates,'paired':paired,'decision':decision}),flush=True)

if __name__=='__main__':main()
