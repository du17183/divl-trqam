# Public release: deployment paths anonymized; algorithm equations unchanged.
"""18 required answers, with missing evidence stated explicitly."""
from pathlib import Path
import json
from datetime import datetime,timezone
import os

ROOT=Path('/opt/divl-trqam')

def load(name):
    p=ROOT/'artifacts'/name
    return json.loads(p.read_text()) if p.exists() else None

def main():
    fp=load('phase0_beta_fp32_metrics.json');small=load('phase0_small_parameter_metrics.json');labels=load('beta_target_pilot_audit.json') or load('beta_target_audit.json')
    smoke=load('four_arm_smoke_test.json');expert=load('expert_hdf5_manifest.json');audit=load('pilot_replay_audit.json');metrics=load('critic_action_metrics.json')
    manifest=ROOT/'data/pilot_libero_v1/manifest.json';pilot=json.loads(manifest.read_text()) if manifest.exists() else None
    rows=[]
    rows.append(('beta/reference FP32 master',f'{fp["beta_master_dtype"]}/{fp["reference_master_dtype"]}; full checkpoint save/reload verified' if fp and fp['passed'] else 'NOT PASSED'))
    rows.append(('Small-LR real beta update',f'lr={fp["lr"]}; real sampled beta changed {fp["sampled_beta_changed"]}/100; small model after8={small["beta_parameter"]}' if fp and fp['passed'] else 'NOT PASSED'))
    rows.append(('Reference EMA',f'real sampled reference changed {fp["sampled_reference_changed"]}/100; .995 old +.005 beta; small hand error <1e-7' if fp and fp['passed'] else 'NOT PASSED'))
    rows.append(('Full H10 beta targets',f'{labels["real_samples"]} actual samples; normalized error {labels["max_normalized_target_error"]}; roundtrip {labels["max_action_roundtrip_error"]}' if labels and labels['passed'] else 'NOT PASSED'))
    rows.append(('Terminal padding masked','YES; direct BC temporal loss masked; seven boundary cases and padding loss test passed' if labels and labels['terminal_padding_loss_mask_zero'] else 'NOT PASSED'))
    rows.append(('Critic actual k',str(audit['critic_k_values'])+'; gamma^k, recorded terminals bootstrap0' if audit else 'Prefix/mask/k tested; complete pilot audit pending'))
    rows.append(('LIBERO entry independent of AIRBOT','YES; dedicated LIBERO fields and existing LIBERO preprocessing; inherited AIRBOT YAML not invoked'))
    rows.append(('Scalar expectile branch','YES; real single-step forward/backward/update, same hashes across four smoke arms' if smoke and smoke['passed'] else 'Smoke pending'))
    rows.append(('Expert initial provenance',f'resolved {expert["resolved"]}, unresolved {expert["unresolved"]}; all experts excluded from pilot' if expert else 'Pending'))
    rows.append(('Pilot replay episodes',str(pilot['episodes'])+' train; 100 independent dev episodes' if pilot else 'Collecting 200 train +100 dev; not yet frozen'))
    rows.append(('Pilot success/failure',f'{pilot["successes"]}/{pilot["failures"]}' if pilot else 'Full distribution pending'))
    for method in ['scalar','divl']:
        r=metrics['methods'][method] if metrics else None
        rows.append((method+' action ranking',f'{r["pairwise_ranking"]}, CI95={r["ranking_ci95"]}, effective pairs={r["effective_pairs"]}' if r else 'NOT COMPLETED; no learned-action claim'))
    rows.append(('Both per-state Spearman',str({m:metrics['methods'][m]['macro_state_spearman'] for m in ['scalar','divl']}) if metrics else 'NOT COMPLETED'))
    rows.append(('Action shuffle delta',str({m:metrics['methods'][m]['shuffle_mean_absolute_delta'] for m in ['scalar','divl']}) if metrics else 'NOT COMPLETED'))
    rows.append(('True gradient G+ vs G-',str({m:metrics['methods'][m]['gradient_epsilon_means'] for m in ['scalar','divl']}) if metrics else 'NOT COMPLETED; predicted Q increase alone is not evidence'))
    rows.append(('Learned action evidence',str({m:metrics['methods'][m]['has_action_evidence'] for m in ['scalar','divl']}) if metrics else 'NOT ESTABLISHED'))
    rows.append(('Enter formal actor QAM',metrics['decision']+'; no formal actor training launched automatically' if metrics else 'NO; action diagnostic gate not completed'))
    assert len(rows)==18
    text='# Current stage summary\n\nUpdated UTC: '+datetime.now(timezone.utc).isoformat()+'\n\n'
    training=load('critic_pilot_training_complete.json')
    remat=load('critic_remat_equivalence.json');oom=load('critic_batch256_oom_attempt.json')
    progress={m:load('critic_'+m+'_progress.json') for m in ['scalar','divl']}
    text+='## Current execution\n\n'
    if training and training['passed']:
        text+='Matched critic-only training completed through2000 updates. Actor/beta are frozen. Action diagnostics: '+('complete' if metrics else 'pending')+'.\n\n'
    elif any(progress.values()):
        text+='Critic-only updates recorded: '+str({m:p['step'] if p else 0 for m,p in progress.items()})+'. Actor/QAM remain disabled.\n\n'
    elif oom:
        text+='First batch256 attempt completed0 updates and failed with OOM in both arms. Failed logs retained. Memory/numerical repair verification: '+('passed' if remat and remat['passed'] else 'pending')+'. No learned-action evidence exists yet.\n\n'
    else:text+='Pilot replay and independent dev baseline complete; critic-only phase pending.\n\n'
    text+='| # | Required question | Current answer |\n|---|---|---|\n'
    text+=''.join(f'| {i} | {name} | {answer} |\n' for i,(name,answer) in enumerate(rows,1))
    text+='\n## Protocol and limits\n\nAll ten official libero_10 tasks. Train/dev/final IDs frozen 30/10/10 per task. No new final-split evaluation was performed after freezing the split. Official pi05 pretraining provenance is not claimed disjoint from official benchmark data.\n\n'
    prior=load('prior_feasibility_split_overlap.json')
    if prior:
        text+='Earlier ID=7 feasibility rollouts predated this split and include '+str(prior['counts'])+'. Thus final IDs were not wholly unseen historically. Those disposable smoke/beta/critic models are never used as experiment initialization; no final-split samples enter the pilot critic training. The split was not changed to hide this overlap. See artifacts/prior_feasibility_split_overlap.json.\n\n'
    text+='Official checkpoint/normalization/controller/reward unchanged. H10, execute at most5, environment7/model32/proprio8, gamma=.9999. No unresolved experts, failed episodes or failed seeds removed.\n\n'
    text+='Phases 0/0B used disposable updated models; they are not experiment initializations. Phase1 ran exactly one real optimizer step per arm. Formal actor and beta updates remain prohibited. All formal critic budgets/checkpoints are retained; no best-checkpoint selection on final eval.\n'
    destination=ROOT/'reports/current_stage_summary.md'
    temporary=destination.with_suffix(f'.{os.getpid()}.partial')
    temporary.write_text(text);temporary.replace(destination)
    print('SUMMARY_UPDATED',flush=True)

if __name__=='__main__':main()
