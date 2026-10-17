"""Dynamic beta: online BC learner with EMA reference update for QAM.

Instead of keeping the BC reference frozen throughout QAM training, this module
maintains an independently trained "beta" model that tracks the recorded-action
distribution via flow-matching, then slowly blends it into the reference used
for QAM path sampling via exponential moving average.

Each QAM step uses the *previous* reference for path generation. After the QAM
actor update, beta_step() runs one BC gradient step on the online beta and then
updates: reference = (1 - ema_weight) * reference + ema_weight * beta.
"""

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import pi0
from openpi.training.precision import ensure_fp32_master, assert_fp32_master, reference_ema


@dataclasses.dataclass(frozen=True)
class DynamicBetaConfig:
    ema_weight: float = 0.005
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    def __post_init__(self):
        if not 0 < self.ema_weight < 1:
            raise ValueError("ema_weight must be in (0, 1)")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


def bc_targets(batch_critic, model_action_dim):
    """Full future behavior actions in ACTOR normalization coordinates.

    Never infer future BC labels from the critic's executed-prefix mask.
    The replay builder must provide independently assembled behavior targets.
    """
    actions = batch_critic["actor_bc_executed_sequence"]
    mask = batch_critic["actor_bc_mask"]
    if actions.ndim != 3 or mask.shape != actions.shape[:2] or actions.shape[-1] > model_action_dim:
        raise ValueError("Invalid full-horizon beta target/mask")
    return jnp.pad(actions, ((0, 0), (0, 0), (0, model_action_dim - actions.shape[-1])))


def initialize_dynamic_beta(
    reference: pi0.Pi0,
    *,
    learning_rate_fn,
    beta_config: DynamicBetaConfig = DynamicBetaConfig(),
) -> tuple[pi0.Pi0, nnx.Optimizer]:
    """Create the online beta model and its optimizer from the current reference.

    The beta model is cloned from the reference and trained with BC. Its
    optimizer uses the same LR schedule as the actor (passed as learning_rate_fn).
    """
    ensure_fp32_master(reference)
    beta = nnx.clone(reference)
    assert_fp32_master(beta)
    optimizer = nnx.Optimizer(
        beta,
        optax.chain(
            optax.clip_by_global_norm(beta_config.max_grad_norm),
            optax.adamw(learning_rate_fn, weight_decay=beta_config.weight_decay),
        ),
        wrt=nnx.Param,
    )
    return beta, optimizer


@nnx.jit
def beta_step(
    beta: pi0.Pi0,
    reference: pi0.Pi0,
    optimizer: nnx.Optimizer,
    observation,
    actions,
    rng: jax.Array,
    *,
    ema_weight: float = 0.005,
    action_mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """One BC update on beta, then EMA-blend into the reference.

    QAM uses the *old* reference this step; this update prepares the next step's.
    """
    assert_fp32_master(beta)
    assert_fp32_master(reference)
    beta.train()

    def loss_fn(model):
        losses = model.compute_loss(rng, observation, actions, train=True)
        if action_mask is None:
            return jnp.mean(losses)
        if losses.shape != action_mask.shape:
            raise ValueError("BC loss must return [B,H] for the complete temporal mask")
        valid = action_mask.astype(jnp.float32)
        return jnp.sum(jnp.where(action_mask, losses, 0.0)) / jnp.maximum(valid.sum(), 1.0)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, optimizer.wrt))(beta)
    norm = optax.global_norm(grads)
    optimizer.update(grads)
    reference_ema(reference, beta, ema_weight)
    beta.eval()
    reference.eval()
    return {"beta/bc_loss": loss, "beta/grad_norm": norm}
