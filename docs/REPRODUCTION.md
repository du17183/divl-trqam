# Installation and reproduction scope

## 1. Core library

This repository contains an OpenPI working-tree snapshot with the DIVL/LIBERO adaptations used by the experiment.
The source repository commit is recorded in `provenance/source_manifest.json`; that commit alone does not include
all local modifications. The manifest records the actual source bytes and publication bytes separately.

Use Linux and a CUDA-compatible JAX environment. The preserved `pyproject.toml` and `uv.lock` describe the
OpenPI environment. `uv sync --frozen` installs the default project dependencies. Large upstream dependencies
may require additional system packages. The publication checks do not establish a fresh GPU environment installation.

The code uses full-parameter FP32 π0.5, not a small standalone MLP. Download and license model assets independently.
No checkpoint, tokenizer weight, raw rollout or expert dataset is bundled here.

## 2. Mathematical implementation

`src/openpi/training/trqam.py` exposes the exact mathematical body of the experiment's
`pilot_actor_math_v1.py`: `tr_path`, `tr_local_loss`, `tr_dual` and `tr_step`.
TRQAM-derived portions retain the original MIT license. The VLA H10 layout, real action dimensions,
min DoubleQ and pretrained IQL/DIVL critic integration are adaptations rather than claims of an official reproduction.

`tr_dual` uses a parameter floor of 0.01 and an effective multiplier of 3, producing an effective floor of 0.03.
The scalar path-KL budget is 1.0. See the source for horizon normalization and the exact solver grid.

## 3. Archived experiment entry points

```
archive/libero_long/
  scripts/                                      # replay, critic, math and native reset helpers
  experiments/rl_pilot_sft5_step1000_8gpu/code/   # original Actor/Critic/native evaluation helpers
  experiments/lwd_expert_rollout_mix_v1/code/    # mixed data training and evaluation
  experiments/*/configs/                       # scientific protocol examples
```

Main mixed-data entry points are `mixed_critic_train_v1.py`, `mixed_actor_train_v1.py`,
`mixed_eval_inference_v1.py`, `mixed_eval_shard_v1.py` and `mixed_eval_audit_v1.py`.
Their original local import structure is preserved. These scripts also verify private data manifests,
checkpoint provenance and saved numerical gates. Those inputs are not supplied by a code-only release.

Server-specific paths have been replaced with public examples:

| Public example | Meaning |
| --- | --- |
| `/opt/divl-trqam` | installed source repository and Python environments |
| `/data/divl-trqam/benchmark` | experiment workspace |
| `/data/divl-trqam/performance2` | fast checkpoint/data volume |
| `/data/divl-trqam/capacity2` | archival checkpoint volume |

Adapt these examples to your own machine. The archived resource coordinators and resume wrappers encode historical
process/proof contracts. They must not be launched against an unrelated server or treated as a public scheduler.
Re-establish current process identities and create new data/provenance manifests for a new experiment.
Published source hashes are not replacements for the old scientific checkpoint/data audit records.

## 4. Evaluation protocol

Use official LIBERO-Long tasks and initial states with the pinned native reset/action semantics.
The original LIBERO dependency commit was `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
Simulation dependencies were installed in a separate environment from JAX training.
Install LIBERO and its MuJoCo/robosuite/robomimic dependencies according to its upstream instructions;
the original requirements are retained under `examples/libero/`.

Each checkpoint/variant has 500 native episodes, 50 for each of 10 tasks. Only complete, audited evaluations
enter aggregate statistics. Match environment/policy seeds and official initial states for paired comparisons.
Report success, return, length and recovery/regression per task. Success CI uses 10 equal-weight tasks and
physical-source clustered bootstrap with 10000 replicates; nominal multiple comparisons were not corrected.

The reported run uses one SFT initialization and one training seed. Fixed official initial states were repeatedly
used for development. Do not describe these results as unseen final-test performance or universal superiority.

The archived local IPC authentication example is `DIVL_TRQAM_IPC_KEY`; use a fresh value for any new deployment.
