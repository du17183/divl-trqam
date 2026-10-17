"""Native pi05 flow law with explicit real action mask and accumulation after one global RNG draw."""
from pathlib import Path
import dataclasses
import importlib.util

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as model_api
from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi.training import optimizer
from openpi.training import weight_loaders

ROOT = Path(__file__).resolve().parents[1]
BASE = '/home/zwl/openpi/.cache/openpi/openpi-assets/checkpoints/pi05_base/params'


def native_training_module():
    import benchmark_native_init_v1 as module
    return module


def make_config(batch_size, fsdp, seed=0):
    return training_config.TrainConfig(
        name='pi05_libero_long_native_masked_v1', exp_name='independent_benchmark',
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, dtype='float32'),
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=20000, decay_lr=5e-6),
        optimizer=optimizer.AdamW(b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.),
        # Keep raw SFT checkpoints for independent selection. B2 maintains its own BC EMA later.
        ema_decay=None, freeze_filter=nnx.Nothing, seed=seed, batch_size=batch_size,
        fsdp_devices=fsdp, num_train_steps=20000, wandb_enabled=False,
        checkpoint_base_dir=str(ROOT / 'checkpoints/sft'))


def as_batch(data):
    observation = model_api.Observation(images=data['images'], image_masks=data['image_masks'], state=data['state'],
                                        tokenized_prompt=data['tokenized_prompt'], tokenized_prompt_mask=data['tokenized_prompt_mask'])
    return observation, data['actions'], data['action_mask']


def prepare(rng, batch, train=True):
    observation, actions, mask = batch
    pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_api.preprocess_observation(pre_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
    return observation, actions, mask, noise, time


def masked_loss(model, observation, actions, mask, noise, time):
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    u_t = noise - actions
    velocity = model.predict_velocity(observation, x_t, time)
    squared = jnp.square(velocity.astype(jnp.float32) - u_t.astype(jnp.float32))
    per_sample = jnp.sum(squared * mask, axis=(-1, -2)) / jnp.maximum(jnp.sum(mask, axis=(-1, -2)), 1.)
    return jnp.mean(per_sample)


def gradients(config, state, prepared, micro_batch):
    """Sum micro gradients before one clip/AdamW update; no per-micro optimizer or RNG changes."""
    global_size = prepared[1].shape[0]
    assert global_size % micro_batch == 0
    count = global_size // micro_batch
    if count == 1:
        model = nnx.merge(state.model_def, state.params)
        model.train()
        return nnx.value_and_grad(masked_loss, argnums=nnx.DiffState(0, config.trainable_filter))(model, *prepared)
    chunks = jax.tree.map(lambda x: x.reshape((count, micro_batch) + x.shape[1:]) if x is not None else None, prepared)
    zero_grads = jax.tree.map(jnp.zeros_like, state.params.filter(config.trainable_filter))
    def body(carry, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()
        loss, grad = nnx.value_and_grad(masked_loss, argnums=nnx.DiffState(0, config.trainable_filter))(model, *batch)
        return jax.tree.map(lambda a, b: a + b / count, carry, grad), loss
    grads, losses = jax.lax.scan(body, zero_grads, chunks)
    return jnp.mean(losses), grads


def make_step(config, micro_batch):
    def step(rng, state, batch):
        prepared = prepare(jax.random.fold_in(rng, state.step), batch)
        loss, grads = gradients(config, state, prepared, micro_batch)
        params = state.params.filter(config.trainable_filter)
        updates, opt_state = state.tx.update(grads, state.opt_state, params)
        model = nnx.merge(state.model_def, state.params)
        nnx.update(model, optax.apply_updates(params, updates))
        new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state)
        stats = {'loss': loss, 'grad_norm': optax.global_norm(grads), 'update_norm': optax.global_norm(updates),
                 'parameter_norm': optax.global_norm(new_state.params.filter(config.trainable_filter))}
        return new_state, stats
    return step
