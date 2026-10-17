"""Native complete-batch transactions. No accumulation and no actor objects."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parameter_digest(tree):
    import jax
    import numpy as np
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(jax.device_get(leaf))
        h.update(str(value.shape).encode() + str(value.dtype).encode() + value.tobytes())
    return h.hexdigest()


def validate_batch(batch, size):
    import numpy as np
    assert batch['actions'].shape == (size, 5, 7), 'No micro batches or changed actions'
    assert batch['action_mask'].shape == (size, 5)
    assert np.asarray(batch['action_mask']).all(), 'Planned fixed H5 mask only'
    for name in ['rewards', 'discounts', 'critic_k']:
        assert batch[name].shape == (size,), name
    k = np.asarray(batch['critic_k'])
    assert ((k >= 1) & (k <= 5)).all()


def make_bundle(config, *, encoder=None):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from critic_model_v1 import build_pretrained, heads, initialize_training
    critic = build_pretrained(config) if encoder is None else heads(config, encoder)
    target, vopt, qopt = initialize_training(critic, config.updates, learning_rate=config.learning_rate)
    bundle = nnx.Dict(critic=critic, target=target, vopt=vopt, qopt=qopt)
    assert all(x.dtype == np.dtype('float32') for x in jax.tree.leaves(nnx.state(critic, nnx.Param)))
    return bundle


def bind_update(bundle, config, *, batch_size=256, fixture=False):
    """Bind after restore; model is always an explicit argument to native nnx.jit."""
    from critic_model_v1 import critic_step
    assert batch_size == 256 or fixture, 'Scientific global batch is frozen at256'

    def update(batch):
        validate_batch(batch, batch_size)
        # One native transaction over ALL rows; never import the rejected microbatch code.
        return critic_step(bundle.critic, bundle.target, bundle.vopt, bundle.qopt, batch, config=config)

    return update


def replicated_layout(bundle):
    """Candidate data-parallel layout; must separately pass full-VLM GPU equivalence."""
    import flax.nnx as nnx
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(np.asarray(jax.devices()), ('critic_batch',))
    replicated = NamedSharding(mesh, PartitionSpec())
    data = NamedSharding(mesh, PartitionSpec('critic_batch'))
    state = jax.tree.map(lambda x: jax.device_put(x, replicated), nnx.state(bundle))
    nnx.update(bundle, state)
    jax.block_until_ready(state)
    return data


def strict_upstream(*, scientific):
    """Fail before importing JAX or loading VLM when data/science gates are not ready."""
    from critic_state_v1 import verified_prerequisites, read, digest
    prereq = verified_prerequisites(require_execution=scientific)
    assert read('metrics/critic_fullbatch_cpu_gate001.json')['passed']
    assert read('metrics/critic_state_gate001.json')['passed']
    # Existing SFT must have exited naturally before profiling competes for its GPUs.
    assert read('metrics/sft_stage002_complete.json'), 'Wait SFT supervisor natural completion'
    failed = read('metrics/sft_stage002_failed.json')
    assert not failed, 'Preserve SFT failure, no downstream launch'
    if scientific:
        gate = read('metrics/critic_pretrained_gpu_gate001.json')
        assert gate['passed'] and gate['full_pretrained_encoder'] and gate['real_policy_replay']
        assert set(gate['value_arms']) == {'scalar', 'divl'}
        assert gate['protocol_sha256'] == prereq['protocol_sha256']
        assert gate['replay_manifest_sha256'] == prereq['replay_manifest_sha256']
        assert gate['execution_sources_sha256'] == prereq['execution_sources_sha256']
        assert prereq['execution']['pretrained_gate_sha256'] == digest(ROOT/'metrics/critic_pretrained_gpu_gate001.json')
        assert prereq['execution']['gradient_equivalence_passed']
        assert prereq['execution']['full_state_restore_jit_passed']
    return prereq


def input_order_digest(draws):
    return hashlib.sha256(json.dumps(draws, separators=(',', ':')).encode()).hexdigest()
