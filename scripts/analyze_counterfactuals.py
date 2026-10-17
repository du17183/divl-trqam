# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Actual return ordering, clustered uncertainty and explicit actor-training gate."""
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path('/opt/divl-trqam')
TIE_TOLERANCE=1e-6

def json_safe(value):
    """Undefined statistics are null, never nonstandard JSON NaN/Infinity."""
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value

def dump(value):
    return json.dumps(json_safe(value),indent=2,allow_nan=False)

def metric(rows):
    pairs=sum(r['effective_pairs'] for r in rows)
    rank=sum(r['correct_pairs'] for r in rows)/pairs if pairs else float('nan')
    spearman=[r['spearman'] for r in rows if r['spearman'] is not None]
    return rank,float(np.mean(spearman)) if spearman else float('nan'),float(np.mean([r['gradient_delta'] for r in rows]))

def budget_metrics(groups,method,steps):
    """Reuse exactly the same real returns for every retained training budget."""
    static=json.loads((ROOT/f'artifacts/static_action_{method}_step{steps}.json').read_text())
    records={row['state_id']:row for row in static['records']};rows=[]
    for group in groups:
        candidates=sorted([r for r in group['branches'] if r['kind']=='candidate'],key=lambda r:r['candidate_index'])
        row=records[group['metadata']['state_id']]
        with np.load(row['path'],allow_pickle=False) as f:
            actions=f['candidate_actions'];q=f['candidate_q']
        np.testing.assert_array_equal(actions,np.asarray([r['action'] for r in candidates],np.float32))
        g=np.asarray([r['future_return'] for r in candidates]);count=correct=0
        for a,b in itertools.combinations(range(8),2):
            if abs(g[a]-g[b])<=TIE_TOLERANCE:continue
            count+=1
            correct+=.5 if abs(q[a]-q[b])<=1e-10 else float((q[a]-q[b])*(g[a]-g[b])>0)
        correlation=float(spearmanr(q,g).statistic) if g.std()>TIE_TOLERANCE and q.std()>1e-10 else None
        rows.append({'task_id':group['metadata']['task_id'],'effective_pairs':count,'correct_pairs':correct,'spearman':correlation,'gradient_delta':0.})
    observed=metric(rows);rng=np.random.default_rng(20261007);draws=[]
    for _ in range(5000):
        sample=[]
        for task in range(10):
            subset=[r for r in rows if r['task_id']==task]
            sample.extend(subset[i] for i in rng.integers(len(subset),size=len(subset)))
        draws.append(metric(sample)[:2])
    draws=np.asarray(draws);ci=[]
    for j in range(2):
        finite=draws[:,j][np.isfinite(draws[:,j])]
        ci.append(np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None])
    return {'steps':steps,'pairwise_ranking':float(observed[0]) if np.isfinite(observed[0]) else None,'ranking_ci95':ci[0],
            'macro_state_spearman':float(observed[1]) if np.isfinite(observed[1]) else None,'spearman_ci95':ci[1],
            'effective_pairs':sum(r['effective_pairs'] for r in rows),'rho':static['rho'],
            'shuffle_mean_absolute_delta':static['shuffle_mean_absolute_delta'],'gradient_norm_mean':static['gradient_norm_mean'],
            'scope':'Same 400 candidate branch outcomes at all budgets; no checkpoint selection. True gradient branches are preregistered at update2000 only.'}

def main():
    groups=json.loads((ROOT/'artifacts/counterfactual_results.json').read_text());assert len(groups)==50
    flattened=[r for g in groups for r in g['branches']]
    pd.DataFrame(flattened).to_parquet(ROOT/'artifacts/counterfactual_results.parquet',index=False)
    results={};method_draws={}
    for method in ['scalar','divl']:
        rng=np.random.default_rng(20261007)
        states=[]
        for group in groups:
            branches=group['branches'];cand=sorted([r for r in branches if r['kind']=='candidate'],key=lambda r:r['candidate_index'])
            assert len(cand)==8
            q=np.asarray([r['q_'+method] for r in cand]);g=np.asarray([r['future_return'] for r in cand])
            count=0;correct=0
            for a,b in itertools.combinations(range(8),2):
                if abs(g[a]-g[b])<=TIE_TOLERANCE:continue
                count+=1
                if abs(q[a]-q[b])<=1e-10:correct+=.5
                elif (g[a]-g[b])*(q[a]-q[b])>0:correct+=1
            correlation=float(spearmanr(q,g).statistic) if g.std()>TIE_TOLERANCE and q.std()>1e-10 else None
            differences={}
            for eps in [.01,.03,.05]:
                pair=[r for r in branches if r['kind']=='gradient' and r['method']==method and r['epsilon']==eps]
                assert len(pair)==2
                differences[str(eps)]=next(r['future_return'] for r in pair if r['direction']==1)-next(r['future_return'] for r in pair if r['direction']==-1)
            states.append({'state_id':group['metadata']['state_id'],'task_id':group['metadata']['task_id'],'initial_state_id':group['metadata']['initial_state_id'],
                           'effective_pairs':count,'correct_pairs':correct,'spearman':correlation,'future_variance':float(g.var()),
                           'gradient_delta':float(np.mean(list(differences.values()))),'gradient_by_epsilon':differences})
        observed=metric(states);bootstrap=[]
        # One state per dev initial ID; sampling whole states keeps action branches clustered.
        # Stratify by task; never bootstrap individual transitions or branch action pairs.
        for _ in range(5000):
            sample=[]
            for task in range(10):
                subset=[r for r in states if r['task_id']==task]
                sample.extend(subset[i] for i in rng.integers(len(subset),size=len(subset)))
            bootstrap.append(metric(sample))
        draws=np.asarray(bootstrap);method_draws[method]=draws;ci=[]
        for j in range(3):
            finite=draws[:,j][np.isfinite(draws[:,j])]
            ci.append(np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None])
        static=json.loads((ROOT/f'artifacts/static_action_{method}_step2000.json').read_text())
        shuffle_ratio=static['shuffle_mean_absolute_delta']/max(np.sqrt(static['total_q_variance']),1e-12)
        coverage=sum(r['effective_pairs']>0 for r in states)
        total_pairs=sum(r['effective_pairs'] for r in states)
        valid_spearman=sum(r['spearman'] is not None for r in states)
        coverage_ok=coverage>=25 and total_pairs>=200 and valid_spearman>=25 and float(np.isfinite(draws[:,:2]).all(axis=1).mean())>=.95
        passed=bool(coverage_ok and np.isfinite(observed).all() and observed[0]>=.60 and ci[0][0]>.50 and ci[1][0]>0 and ci[2][0]>0 and shuffle_ratio>.01)
        result={'pairwise_ranking':float(observed[0]) if np.isfinite(observed[0]) else None,'ranking_ci95':ci[0],
                'macro_state_spearman':float(observed[1]) if np.isfinite(observed[1]) else None,'spearman_ci95':ci[1],
                'true_gradient_plus_minus_return_delta':float(observed[2]),'gradient_delta_ci95':ci[2],
                'effective_pairs':sum(r['effective_pairs'] for r in states),'states_with_effective_pairs':sum(r['effective_pairs']>0 for r in states),
                'states':50,'tie_tolerance':TIE_TOLERANCE,'mean_action_variance':static['mean_action_variance'],'rho':static['rho'],
                'shuffle_mean_absolute_delta':static['shuffle_mean_absolute_delta'],'shuffle_relative_q_std':shuffle_ratio,
                'gradient_norm_mean':static['gradient_norm_mean'],'gradient_epsilon_means':{str(e):float(np.mean([r['gradient_by_epsilon'][str(e)] for r in states])) for e in [.01,.03,.05]},
                'has_action_evidence':passed,'per_state':states}
        result['coverage_gate']={'passed':coverage_ok,'minimum_states_with_pairs':25,'minimum_pairs':200,'minimum_states_with_spearman':25,
            'finite_bootstrap_fraction':float(np.isfinite(draws[:,:2]).all(axis=1).mean()),
            'reason':'Preregistered before branch outcomes: a handful of informative states cannot establish suite-level evidence.'}
        result['per_task']={str(task):{'states':sum(r['task_id']==task for r in states),
            'effective_pairs':sum(r['effective_pairs'] for r in states if r['task_id']==task),
            'metrics':json_safe(list(metric([r for r in states if r['task_id']==task])))} for task in range(10)}
        results[method]=result
        (ROOT/f'reports/critic_{method}_report.md').write_text(f'# {method} critic\n\n'+dump({k:v for k,v in result.items() if k!='per_state'})+'\n\n'
            'These are dev diagnostic states, never final eval. Updates were critic-only; actor/beta frozen. All training checkpoints are retained.\n')
    paired={}
    for index,name in enumerate(['ranking','spearman','true_gradient_return_delta']):
        differences=method_draws['divl'][:,index]-method_draws['scalar'][:,index]
        differences=differences[np.isfinite(differences)]
        paired[name+'_divl_minus_scalar_ci95']=np.quantile(differences,[.025,.975]).tolist() if len(differences) else [None,None]
    budgets={method:[budget_metrics(groups,method,step) for step in [250,500,1000,2000]] for method in ['scalar','divl']}
    output={'methods':results,'all_budgets':budgets,'paired_comparison':paired,'protocol':'50 dev initial-state clusters, 8 candidate actions, 3 epsilon +/- directions per critic; gamma=.9999; same frozen policy continuation noise',
            'bootstrap':'5000 resamples of whole initial-state clusters, stratified by task','decision':'EVIDENCE_FOR_NEXT_STAGE' if any(r['has_action_evidence'] for r in results.values()) else 'STOP_ACTOR_QAM',
            'gate_note':'Engineering reference ranking .60, positive clustered CI, positive Spearman and true gradient direction, meaningful shuffle; at least25 informative states and200 pairs. Preregistered before outcomes; not a paper significance threshold. No formal actor training is started automatically.'}
    (ROOT/'artifacts/critic_action_metrics.json').write_text(dump(output))
    table='| Method | Ranking | Spearman | True G+ minus G- | Gate |\n|---|---:|---:|---:|---|\n'
    for m,r in results.items():table+=f'| {m} | {r["pairwise_ranking"]} | {r["macro_state_spearman"]} | {r["true_gradient_plus_minus_return_delta"]} | {r["has_action_evidence"]} |\n'
    text='# Learned critic action diagnostic\n\n'+table+'\nDecision: '+output['decision']+'\n\n'
    text+='All 400 candidate branches and 600 gradient branches are preserved. Return ties are excluded from ranking and their count is reported. Static gradient/nonzero Q alone does not pass the gate.\n'
    text+='\n## All retained training budgets\n\n| Method | Updates | Ranking | Spearman | Action shuffle delta |\n|---|---:|---:|---:|---:|\n'
    for method,rows in budgets.items():
        for row in rows:text+=f'| {method} | {row["steps"]} | {row["pairwise_ranking"]} | {row["macro_state_spearman"]} | {row["shuffle_mean_absolute_delta"]} |\n'
    (ROOT/'reports/critic_action_diagnostic.md').write_text(text)
    (ROOT/'reports/counterfactual_branch_report.md').write_text('# Same-state counterfactual branches\n\n'+text+'\nSnapshot, controller, warm-start physics and RNG restored; initial-state and proprio error recorded for every branch. Environment reconstruction was audited separately before collection.\n')
    print(dump({m:{k:v for k,v in r.items() if k!='per_state'} for m,r in results.items()}),flush=True)

if __name__=='__main__':main()
