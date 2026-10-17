# Public release: deployment paths anonymized; algorithm equations unchanged.
"""CPU-only checks of real LWD operators and collected LIBERO boundaries."""
import json
from pathlib import Path
import traceback

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms
from openpi.models import divl_losses, pi0_config
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.shared import normalize
from openpi.training import beta_training, qam
from openpi.training.qam_training import QAMTrainingConfig
from openpi.training.qam_libero_adapter import LiberoQAMBatchBuilder

ROOT = Path("/opt/divl-trqam")


class QuadraticHead:
    def __call__(self, z, actions, mask):
        value = -jnp.sum((actions - 0.6) ** 2 * mask[..., None], axis=(1, 2))
        return value, value + 0.1


class ZeroFlow:
    def predict_velocity(self, *, observation, noisy_actions, time):
        return jnp.zeros_like(noisy_actions)


class TinyBeta(nnx.Module):
    def __init__(self, dtype):
        self.action_dim = 32
        self.weight = nnx.Param(jnp.ones((1,), dtype=dtype))

    def compute_loss(self, rng, observation, actions, train):
        return (self.weight.value - actions) ** 2


class ShapeTokenizer:
    """Only for batch-shape testing; not used as a trained critic tokenizer."""
    def tokenize(self, prompt, state):
        assert state.shape == (8,)
        return np.zeros(64, dtype=np.int32), np.ones(64, dtype=bool)


def main():
    checks, findings = [], []
    def check(name, fn):
        try:
            details = fn()
            checks.append({"name": name, "passed": True, "details": details})
        except Exception as error:
            checks.append({"name": name, "passed": False, "error": repr(error), "traceback": traceback.format_exc()})
    atoms = jnp.asarray([0.1, 0.2, 0.5, 0.8, 1.0])
    def quantile():
        logits = jnp.log(jnp.asarray([[0.1, 0.2, 0.3, 0.2, 0.2]]))
        result = np.asarray(divl_losses.categorical_quantile(logits, atoms, jnp.asarray([0.5])))
        np.testing.assert_allclose(result, [0.5])
        return {"manual_quantile": 0.5, "computed": result.tolist()}
    check("categorical_quantile_hand_calculation", quantile)
    def entropy():
        value = divl_losses.adaptive_quantile_level(jnp.zeros((2, 201)), tau_base=0.6, alpha=0.3, tau_min=0, tau_max=1)
        np.testing.assert_allclose(value, [0.3, 0.3], atol=1e-6)
        return {"uniform_entropy_tau": np.asarray(value).tolist()}
    check("entropy_adaptive_tau", entropy)
    def terminal():
        value = divl_losses.divl_q_target(jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 0.0]),
                                          jnp.zeros((2, 5)), atoms, tau_base=0.6, alpha=0.3, tau_min=0, tau_max=1)
        np.testing.assert_allclose(value, [1, 0])
        return {"terminal_target": np.asarray(value).tolist()}
    check("terminal_bootstrap_mask", terminal)
    def projection():
        result = divl_losses.project_scalar_to_atoms(jnp.asarray([0.35]), atoms)
        np.testing.assert_allclose(result.sum(axis=-1), [1])
        np.testing.assert_allclose(result @ atoms, [0.35], atol=1e-6)
        return {"probability_sum": float(result.sum())}
    check("scalar_to_distribution_projection", projection)
    def direction():
        endpoint = jnp.ones((2, 10, 32)) * 0.1
        low, high = -jnp.ones(7), jnp.ones(7)
        critic = nnx.Dict(q_head=QuadraticHead())
        adjoint, metrics = qam.qam_terminal_adjoint(critic, jnp.zeros((2, 1)), endpoint,
            low, high, low, high, temperature=2, replan_steps=5, critic_horizon=10, action_dim=7)
        def score(a):
            physical, mask = qam.actions_for_critic(a, low, high, low, high,
                replan_steps=5, critic_horizon=10, action_dim=7)
            return critic.q_head(None, physical, mask)[0]
        gradient = -2 * adjoint
        random = jax.random.normal(jax.random.key(9), gradient.shape)
        random *= jnp.linalg.norm(gradient, axis=(1, 2), keepdims=True) / jnp.linalg.norm(random, axis=(1, 2), keepdims=True)
        better = score(endpoint + 0.01 * gradient) - score(endpoint)
        baseline = score(endpoint + 0.01 * random) - score(endpoint)
        assert bool(jnp.all(better > 0)) and bool(jnp.all(better > baseline))
        np.testing.assert_allclose(adjoint[:, 5:], 0, atol=1e-7)
        return {"analytic_critic_delta_q": np.asarray(better).tolist(),
                "matched_random_delta_q": np.asarray(baseline).tolist(),
                "scope": "analytic unit test, not learned-critic validation"}
    check("qam_action_gradient_and_executed_prefix", direction)
    def adjoint():
        times = jnp.linspace(0.1, 1, 11)
        path = jnp.ones((11, 2, 10, 32)) * 0.1
        output = qam.qam_backward_adjoint(ZeroFlow(), None, times, path, jnp.ones((2, 10, 32)))
        assert output.shape == path.shape and bool(jnp.isfinite(output).all())
        return {"shape": list(output.shape)}
    check("backward_adjoint_finite_horizon10", adjoint)
    def precision():
        outputs = {}
        for name, dtype in [("bf16", jnp.bfloat16), ("fp32", jnp.float32)]:
            ref = TinyBeta(dtype)
            beta, optimizer = beta_training.initialize_dynamic_beta(ref, learning_rate_fn=2e-5)
            for i in range(8):
                beta_training.beta_step(beta, ref, optimizer, None, jnp.zeros((1, 10, 32)), jax.random.key(i))
            outputs[name] = {"beta_weight": float(beta.weight.value[0]), "reference_weight": float(ref.weight.value[0])}
        assert outputs["bf16"]["beta_weight"] == 1.0
        assert outputs["fp32"]["beta_weight"] < 1.0
        findings.append({"kind": "confirmed_precision_risk", "details": outputs,
                         "scope": "tiny parameter numerical reproduction, not a full pi05 training experiment"})
        return outputs
    check("dynamic_beta_master_precision_reproduction", precision)
    for path in sorted((ROOT / "data/feasibility_libero_long").glob("*.npz")):
        def replay_check(path=path):
            with np.load(path, allow_pickle=False) as f:
                ep = {k: f[k] for k in f.files}
            c = len(ep["rewards"])
            assert len(ep["state"]) == c + 1 and len(ep["base"]) == c + 1
            assert ep["actions"].shape == (c, 10, 7)
            assert ep["actor_bc_executed_sequence"].shape == (c, 10, 7)
            assert ep["discounts"][-1] == 0 and ep["terminal"][-1]
            computed = []
            value = 0.0
            for r, d in zip(ep["rewards"][::-1], ep["discounts"][::-1]):
                value = float(r) + float(d) * value
                computed.append(value)
            np.testing.assert_allclose(computed[::-1], ep["mc_returns"], atol=1e-6)
            norm = normalize.load(ROOT / "checkpoints/pi05_libero/assets/physical-intelligence/libero")
            cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
            batch_cfg = QAMTrainingConfig(critic_horizon=10, replan_steps=5, sample_from_reference=True)
            critic_tokenizer = DIVLTokenizer(DIVLStateEncoderConfig(gemma_tokenizer=str(ROOT / "assets/critic_tokenizer/tokenizer_gemma3.model")))
            builder = LiberoQAMBatchBuilder(cfg, norm, norm, critic_tokenizer, training_config=batch_cfg, td_steps=10)
            batch = builder([(ep, 0)])
            assert batch.actor_observation.state.shape == (1, 32)
            assert batch.critic["actions"].shape == (1, 10, 7)
            current = beta_training.bc_targets(batch.critic, 32)
            correct = transforms.Normalize(norm, use_quantiles=True)({"actions": ep["actor_bc_executed_sequence"][:1]})["actions"]
            correct = np.pad(correct, ((0, 0), (0, 0), (0, 25)))
            discrepancy = float(np.max(np.abs(np.asarray(current) - correct)))
            if discrepancy > 1e-5:
                findings.append({"kind": "confirmed_beta_prefix_label_mismatch", "episode": str(path),
                                 "max_normalized_error": discrepancy})
            return {"path": str(path), "chunks": c, "beta_label_mismatch": discrepancy,
                    "critic_tokenizer_scope": "actual Gemma3 tokenizer; encoder quality not tested"}
        check("collected_replay_and_qam_batch_" + path.stem, replay_check)
    result = {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks),
              "findings": findings, "platform": str(jax.devices()),
              "scope": "CPU mathematics/interface checks; full policy/critic training not performed"}
    (ROOT / "results/libero_lwd_interface_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
