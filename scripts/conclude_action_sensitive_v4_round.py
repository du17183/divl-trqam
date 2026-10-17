"""Close the gated round with actual throughput and bounded interpretation."""
import json,numpy as np,pathlib,collections
from action_sensitive_v4_common import R,D,O,REP,atomic,sha

def main():
 protocol=json.loads((D/'protocol.json').read_text());data=json.loads((D/'manifest.json').read_text())
 audit=json.loads((O/'supervision.json').read_text());probe=json.loads((R/'artifacts/v4_representation_probe.json').read_text())
 status=json.loads((O/'analysis_status.json').read_text());summary=audit['summary'];metrics=probe['probes']
 assert status['stage'] in ['STOP_DATA_REDESIGN','STOP_REPRESENTATION','STOP_CRITIC_ARCHITECTURE','STOP_ACTOR_QAM','ALLOW_FIXED_REFERENCE_QAM_PILOT']
 assert not probe['gate_passed'] and not (R/'checkpoints/critic_replay_v4').exists(), 'This round conclusion records the failed pretraining gate only'
 unchanged=all(sha(p)==h for p,h in protocol['frozen_input_hashes'].items());assert unchanged
 pre=json.loads((O/'branches_preflight.json').read_text())['records'][0]['records'];preids={r['episode_id'] for r in pre}
 new=[r for r in data['records'] if r['episode_id'] not in preids];elapsed=data['elapsed_seconds']
 runtime=dict(inference_gpus=8,sim_workers=16,collection_seconds=elapsed,new_states=data['states']-1,new_branches=len(new),
  states_per_minute=(data['states']-1)*60/elapsed,outcomes_per_minute=len(new)*60/elapsed,
  env_steps_per_second=sum(r['steps'] for r in new)/elapsed,output_bytes_per_second=sum(pathlib.Path(r['path']).stat().st_size for r in new)/elapsed,
  preflight_cached_state_excluded=True,probe_seconds=probe['elapsed_seconds'],gpu_attribution='GPU-wide utilization includes unrelated jobs; our eight inference servers were stopped after collection')
 atomic(O/'runtime_summary.json',runtime)
 def estimate(x):return f'{100*x["mean"]:.2f}% [{100*x["ci95"][0]:.2f}%, {100*x["ci95"][1]:.2f}%]'
 text='# V4 round conclusion\n\n**'+status['stage']+'**. All actor/QAM/dynamic-beta updates remain disabled.\n\n'
 text+='| Probe | Overall ranking /95%CI | Near-ID ranking /95%CI | Overall Spearman |\n|---|---|---|---:|\n'
 names={'A':'Frozen pi05 visual + action','B':'Frozen pi05 visual + proprio + action (primary)','C':'Proprio + action','D':'GT geometry/contact + proprio + action (diagnostic only)'}
 for n in ['A','B','C','D']:
  m=metrics[n]['metrics'];text+=f'| {n}: {names[n]} | {estimate(m["all"]["rank"])} | {estimate(m["Near-ID"]["rank"])} | {m["all"]["spearman"]["mean"]:.4f} |\n'
 text+='\n## Data and fixed gates\n\n'
 text+=f'{data["states"]} states, 30/task, {data["outcomes"]} real branches. Near-ID {summary["near_candidate_ratio"]:.2%}, Mid {summary["support_counts"]["Mid"]/data["outcomes"]:.2%}, Far {summary["support_counts"]["Far-OOD"]/data["outcomes"]:.2%}. Maximum restore return error {summary["baseline_max_error"]}; all2100outcomes retained. {summary["near_effective_states"]} unique held-out Near-ID informative states, exceeding100. Within-Near-ID974pairs:624strictpairs;462with|deltaG|>1e-4,204>1e-3,31>1e-2. Mixed success/failure30/300=10%.\n\n'
 text+='Primary B fails the unchanged overall>=60% and newNear-ID>=65% gates, although its overall Spearman CI is positive and the>=100state coverage gate passes. No formal v4 Scalar/DIVL comparison, gradient branch or Actor/QAM was launched. This is weak learnability evidence, not a demonstration of theoretical unlearnability or that LWD is ineffective.\n\n'
 text+='## Interpretation\n\n'
 diff=probe['paired']['D_minus_B'];ba=probe['paired']['B_minus_A'];cb=probe['paired']['C_minus_B']
 text+=f'1. Coverage improves substantially, but it does not reproduce the old sparseNear-ID67% signal. v3andv4 have different candidates and feature sources; their accuracy difference is descriptive, not a paired treatment estimate. Allstates remain fixed.\n\n'
 text+=f'2. GT−B overall paired difference {diff["all"]["rank"]["mean"]*100:+.3f}pp [ {diff["all"]["rank"]["ci95"][0]*100:+.3f}, {diff["all"]["rank"]["ci95"][1]*100:+.3f} ]pp; Near-ID {diff["Near-ID"]["rank"]["mean"]*100:+.3f}pp CI crosses0. The fixedGTdiagnostic does not show the large representation advantage required forSTOP_REPRESENTATION. It is not a mathematical upper bound: geometry/contact features use PCA32 and omit some hidden dynamics, so this does not exclude all representation bottlenecks.\n\n'
 text+=f'3. Explicit proprio adds only {ba["all"]["rank"]["mean"]*100:+.3f}pp overall, pairedCI crosses0. Removing vision changes ranking by {cb["all"]["rank"]["mean"]*100:+.3f}pp, pairedCI below0. No evidence that visual inputs simply add noise. The formalcritic already receives8Dproprio through normalized/quantizedstate tokens; Bdoes not prove a missing-state adapter bug.\n\n'
 text+='4. Only3.18% of all within-Near-ID pairs have|deltaG|>.01; much of the supervision is a small discounted completion-time difference. Current evidence motivates an audit of first5step intervention horizon, frozen continuation stochasticity and final-return target signal, rather than morecriticupdates or a new largev5collection. Reward, gamma and action/controller definitions remain unchanged.\n\n'
 text+='5. The originalNear-ID measure is a visual-neighborhood action support proxy. Only86/300behavior anchors themselves areNear-ID. Interpolation supplies598/830Near-ID candidates, but its RMSdistance from behavior has median.0748andp90.516; beingNear to another observed action does not establish that it is a tiny localQAM perturbation at the currentphysicalstate. Local-noise/PCA are genuinely small behavior-centered changes but often fall outside that proxy. Audit this distinction before interpreting Near-ID as causal local support.\n\n'
 text+='## Next bounded diagnostic\n\nDo not launchActoror formalScalar/DIVL, do not automaticallyincreaseupdates or collectv5. A next preregistered small train-only diagnostic should separate completion-time-only pairs from larger-return contrasts and test repeat-frozen-continuation variance at identical(s,A). Preserve existingnative reward/gamma/actor/controller and allsamples. This is a recommendation; no additionalbranches were started thisround.\n\n'
 text+='## Actual runtime\n\n'+json.dumps(runtime,indent=2)+'\n\n## Provenance\n\nManifest SHA256`'+sha(D/'manifest.json')+'`; config SHA256`'+sha(D/'protocol.json')+'`. Original data/truth/v3reports hashes unchanged.\n'
 (REP/'v4_round_conclusion.md').write_text(text)
 atomic(O/'round_complete.json',dict(complete=True,stage=status['stage'],states=data['states'],outcomes=data['outcomes'],
   actor_updated=False,formal_critics_trained=(R/'checkpoints/critic_replay_v4').exists(),GT_formal_input=False,original_input_hashes_unchanged=unchanged,
   manifest_sha256=sha(D/'manifest.json'),config_sha256=sha(D/'protocol.json')))
 print('V4_ROUND_COMPLETE',json.dumps(runtime),flush=True)
if __name__=='__main__':main()
