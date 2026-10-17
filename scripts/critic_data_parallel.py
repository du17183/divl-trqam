"""GSPMD data parallelism: global losses/Adam steps stay unchanged."""
import numpy as np
import jax
import flax.nnx as nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


class CriticDataParallel:
    def __init__(self, critic, target, value_optimizer, q_optimizer, devices):
        self.devices = list(devices)
        self.mesh = Mesh(np.asarray(self.devices), ('data',))
        self.replicated = NamedSharding(self.mesh, P())
        self.batched = NamedSharding(self.mesh, P('data'))
        self.bundle = nnx.Dict(critic=critic, target=target,
                               value_optimizer=value_optimizer, q_optimizer=q_optimizer)
        self.restore(nnx.state(self.bundle))

    def restore(self, state):
        nnx.update(self.bundle, jax.tree.map(lambda x: jax.device_put(x, self.replicated), state))

    def batch(self, batch):
        leaves = jax.tree.leaves(batch)
        size = leaves[0].shape[0]
        assert size % len(self.devices) == 0
        assert all(x.ndim and x.shape[0] == size for x in leaves)
        return jax.tree.map(lambda x: jax.device_put(x, self.batched), batch)

    def evidence(self, batch):
        leaf = batch['actions']
        return dict(device_count=len(self.devices), global_batch=int(leaf.shape[0]),
                    per_device_batch=int(leaf.shape[0] // len(self.devices)),
                    devices=[str(d) for d in self.devices],
                    batch_shards=[dict(device=str(x.device), shape=list(x.data.shape))
                                  for x in leaf.addressable_shards],
                    parameter_shard_devices=sorted({str(x.device)
                        for v in jax.tree.leaves(nnx.state(self.bundle.critic, nnx.Param))
                        for x in v.addressable_shards}),
                    loss_reduction='Global batch mean; automatic GSPMD gradient all-reduce',
                    optimizer='One shared Adam/EMA update per global batch')
