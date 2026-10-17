# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Apply narrowly scoped fixes to the owned copy, with exact-context guards."""
from pathlib import Path
ROOT = Path('/opt/divl-trqam')
assert ROOT.resolve() == Path('/DATA/disk1/opt/divl-trqam')


def replace(name, old, new):
    p = ROOT / name
    text = p.read_text()
    assert text.count(old) == 1, (name, old[:100], text.count(old))
    p.write_text(text.replace(old, new))


replace('src/openpi/training/beta_training.py', 'from openpi.models import pi0',
        'from openpi.models import pi0\nfrom openpi.training.precision import ensure_fp32_master, assert_fp32_master, reference_ema')
replace('src/openpi/training/beta_training.py', '''    """Normalized recorded actions with repeated terminal padding, matching BC."""
    actions, mask = batch_critic["actions"], batch_critic["action_mask"]
    lengths = jnp.sum(mask, axis=1).astype(jnp.int32)
    last = actions[jnp.arange(actions.shape[0]), jnp.maximum(lengths - 1, 0)]
    actions = jnp.where(mask[..., None], actions, last[:, None])
    return jnp.pad(actions, ((0, 0), (0, 0), (0, model_action_dim - actions.shape[-1])))''', '''    """Full future behavior actions in ACTOR normalization coordinates.

    Never infer future BC labels from the critic's executed-prefix mask.
    The replay builder must provide independently assembled behavior targets.
    """
    actions = batch_critic["actor_bc_executed_sequence"]
    mask = batch_critic["actor_bc_mask"]
    if actions.ndim != 3 or mask.shape != actions.shape[:2] or actions.shape[-1] > model_action_dim:
        raise ValueError("Invalid full-horizon beta target/mask")
    return jnp.pad(actions, ((0, 0), (0, 0), (0, model_action_dim - actions.shape[-1])))''')
replace('src/openpi/training/beta_training.py', '    beta = nnx.clone(reference)',
        '    ensure_fp32_master(reference)\n    beta = nnx.clone(reference)\n    assert_fp32_master(beta)')
replace('src/openpi/training/beta_training.py', '    ema_weight: float = 0.005,\n) -> dict[str, jax.Array]:',
        '    ema_weight: float = 0.005,\n    action_mask: jax.Array | None = None,\n) -> dict[str, jax.Array]:')
replace('src/openpi/training/beta_training.py', '''    beta.train()

    def loss_fn(model):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))''', '''    assert_fp32_master(beta)
    assert_fp32_master(reference)
    beta.train()

    def loss_fn(model):
        losses = model.compute_loss(rng, observation, actions, train=True)
        if action_mask is None:
            return jnp.mean(losses)
        if losses.shape != action_mask.shape:
            raise ValueError("BC loss must return [B,H] for the complete temporal mask")
        valid = action_mask.astype(jnp.float32)
        return jnp.sum(jnp.where(action_mask, losses, 0.0)) / jnp.maximum(valid.sum(), 1.0)''')
replace('src/openpi/training/beta_training.py', '''    nnx.update(
        reference,
        jax.tree.map(
            lambda old, new: (1 - ema_weight) * old + ema_weight * new,
            nnx.state(reference, nnx.Param),
            nnx.state(beta, nnx.Param),
        ),
    )''', '    reference_ema(reference, beta, ema_weight)')
replace('src/openpi/training/qam_training.py', 'from openpi.training import qam',
        'from openpi.training import qam\nfrom openpi.training.precision import ensure_fp32_master')
replace('src/openpi/training/qam_training.py', '''    Clone the independent reference before changing actor parameter dtypes.
    Reference keeps the loaded precision; actor gets FP32 master parameters.''',
        '''    Promote actor masters BEFORE cloning reference. Both actor and
    independent reference retain FP32 masters; model compute dtype is unchanged.''')
replace('src/openpi/training/qam_training.py', '''    bc_actor.eval()
    reference = nnx.clone(bc_actor)
    parameters = nnx.state(bc_actor, nnx.Param)
    nnx.update(bc_actor, jax.tree.map(lambda value: value.astype(jnp.float32), parameters))''',
        '''    bc_actor.eval()
    ensure_fp32_master(bc_actor)
    reference = nnx.clone(bc_actor)''')
replace('src/openpi/training/qam_training.py', '            ema_weight=cfg.ema_weight,',
        '            ema_weight=cfg.ema_weight,\n            action_mask=batch.critic["actor_bc_mask"],')
replace('src/openpi/training/qam_libero_adapter.py', 'from openpi.training import divl_libero_adapter as divl',
        'from openpi.training import divl_libero_adapter as divl\nfrom openpi.training.libero_replay_contract import validate_episode')
replace('src/openpi/training/qam_libero_adapter.py', '        self.critic_norm_stats = critic_norm_stats',
        '        self.critic_norm_stats = critic_norm_stats\n        self.actor_normalize = transforms.Normalize(actor_norm_stats, use_quantiles=True)')
replace('src/openpi/training/qam_libero_adapter.py', '        actor_inputs = []',
        '        actor_inputs, bc_actions, bc_masks = [], [], []\n        validated = {}')
replace('src/openpi/training/qam_libero_adapter.py', '            actions = episode["actions"][index]', '''            identity = id(episode)
            if identity not in validated:
                validated[identity] = validate_episode(episode)
            full, full_mask = validated[identity]
            bc_actions.append(full[index])
            bc_masks.append(full_mask[index])
            actions = episode["actions"][index]''')
replace('src/openpi/training/qam_libero_adapter.py', '        return QAMBatch(actor_observation=observation, critic=critic_batch)', '''        critic_batch["actor_bc_executed_sequence"] = jnp.asarray(
            self.actor_normalize({"actions": np.stack(bc_actions)})["actions"], dtype=jnp.float32)
        critic_batch["actor_bc_mask"] = jnp.asarray(np.stack(bc_masks), dtype=bool)
        critic_batch["critic_executed_actions"] = critic_batch["actions"]
        critic_batch["critic_action_mask"] = critic_batch["action_mask"]
        critic_batch["critic_k"] = jnp.asarray([int(e["lengths"][i]) for e, i in samples])
        return QAMBatch(actor_observation=observation, critic=critic_batch)''')
print('Phase0/0B precision, independent BC labels and temporal loss masks patched')
