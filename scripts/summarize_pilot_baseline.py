# Public release: deployment paths anonymized; algorithm equations unchanged.
"""All-task frozen dev characterization and honest sparse-return description."""
from pathlib import Path
import json
import numpy as np

ROOT=Path('/opt/divl-trqam')

def main():
    train=json.loads((ROOT/'data/pilot_libero_v1/manifest.json').read_text())
    dev=json.loads((ROOT/'data/dev_libero_v1/manifest.json').read_text())
    stats={};table='| Task | Dev success | Mean env steps | Fail initial IDs |\n|---|---:|---:|---|\n'
    for task in range(10):
        rows=[r for r in dev['records'] if r['task_id']==task]
        stats[str(task)]={'name':rows[0]['task_name'],'episodes':len(rows),'successes':sum(r['success'] for r in rows),
            'success_rate':float(np.mean([r['success'] for r in rows])),'mean_env_steps':float(np.mean([r['environment_steps'] for r in rows])),
            'failure_initial_ids':[r['initial_state_id'] for r in rows if not r['success']]}
        s=stats[str(task)];table+=f'| {task}: {s["name"]} | {s["successes"]}/{s["episodes"]} | {s["mean_env_steps"]:.1f} | {s["failure_initial_ids"]} |\n'
    success=dev['successes']/dev['episodes'];rng=np.random.default_rng(20261007);draws=[]
    for _ in range(5000):
        outcomes=[]
        for task in range(10):
            rows=[r for r in dev['records'] if r['task_id']==task]
            outcomes.extend(rows[i]['success'] for i in rng.integers(len(rows),size=len(rows)))
        draws.append(np.mean(outcomes))
    result={'episodes':dev['episodes'],'successes':dev['successes'],'success_rate':success,'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'mean_env_steps':float(np.mean([r['environment_steps'] for r in dev['records']])),'per_task':stats,
            'ceiling_at_98_percent':success>=.98,'split':'dev only; final eval untouched','scope':'Frozen official pi05 baseline; not performance of any learned critic/actor'}
    (ROOT/'artifacts/frozen_dev_characterization.json').write_text(json.dumps(result,indent=2))
    returns=np.asarray([r['return'] for r in train['records']]);future=[]
    for row in train['records']:
        with np.load(row['path'],allow_pickle=False) as e:future.extend(e['mc_returns'].tolist())
    future=np.asarray(future)
    distribution={'episodes':train['episodes'],'successes':train['successes'],'failures':train['failures'],
                  'episode_return_mean':float(returns.mean()),'episode_return_std':float(returns.std()),'episode_return_quantiles':np.quantile(returns,[0,.25,.5,.75,1]).tolist(),
                  'chunk_future_histogram':np.histogram(future,bins=np.linspace(0,1,21))[0].tolist(),'chunk_future_quantiles':np.quantile(future,[0,.25,.5,.75,1]).tolist(),
                  'interpretation':'Native reward is binary and collection stops at first success. Discounted intermediate nonzero returns mainly encode time to success, not invented partial-progress rewards. A global histogram does not establish conditional action value structure.'}
    (ROOT/'artifacts/pilot_return_distribution.json').write_text(json.dumps(distribution,indent=2))
    text='# Frozen dev baseline\n\n'+table+f'\nOverall {dev["successes"]}/{dev["episodes"]}, {success:.1%}; initial-state bootstrap CI {result["ci95"]}. Final eval unused.\n\n'
    text+=distribution['interpretation']+'\n'
    (ROOT/'reports/frozen_dev_baseline_report.md').write_text(text)
    with (ROOT/'reports/pilot_replay_report.md').open('a') as f:f.write('\n'+text)
    print(json.dumps({'dev_success':success,'dev_ci':result['ci95'],'train_successes':train['successes'],'train_failures':train['failures']},indent=2),flush=True)

if __name__=='__main__':main()
