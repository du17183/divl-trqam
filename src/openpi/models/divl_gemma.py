import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.linen as nn
from functools import partial
from gemma import gm
import jax
import jax.numpy as jnp

from openpi.models.divl_config import DIVLStateEncoderConfig


@partial(nn.remat, prevent_cse=True)
def _checkpoint_block(block, x, positions, attention_mask):
    # Same block parameters/forward; recompute activations for backward rather
    # than retain every full-sequence attention map at batch256.
    return block(x, positions, None, attention_mask)


class _GemmaForCritic(gm.nn.Gemma3_270M):
    def embed_tokens(self, token_ids: jax.Array) -> jax.Array:
        # Reuse the official embedding table and embedding scaling
        return self.embedder.encode(token_ids)

    def encode_embeddings(
        self,
        embeddings: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        *,
        checkpoint_enabled: bool = True,
    ) -> jax.Array:
        x = embeddings

        for block in self.blocks:
            # No KV cache is needed for this full-sequence forward
            if checkpoint_enabled:
                _, x = _checkpoint_block(block, x, positions, attention_mask)
            else:
                _, x = block(x, positions, None, attention_mask)

        return self.final_norm(x)


class DIVLGemmaBackbone(nnx.Module):
    def __init__(self, config: DIVLStateEncoderConfig, *, rngs: nnx.Rngs):
        self.compute_dtype = jnp.dtype(config.compute_dtype)
        self.gradient_checkpointing = True

        self.model = nnx_bridge.ToNNX(_GemmaForCritic(dtype=self.compute_dtype))

        self.model.lazy_init(
            jnp.full((1, 1), 2, dtype=jnp.int32),
            return_last_only=True,
            rngs=rngs,
        )

    def _compute_model(self):
        # Gemma's dtype option controls parameter initialization, not the dtype
        # of its plain jnp.einsum calls. Keep original master storage FP32 and
        # create a differentiable, read-only view for each compute invocation.
        definition, parameters, other = nnx.split(self.model, nnx.Param, ...)
        compute_parameters = jax.tree.map(
            lambda x: x.astype(self.compute_dtype), parameters
        )
        return nnx.merge(definition, compute_parameters, other)

    def embed_tokens(self, token_ids: jax.Array) -> jax.Array:
        return self._compute_model()(token_ids, method="embed_tokens").astype(self.compute_dtype)

    def __call__(
        self,
        embeddings: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        return self._compute_model()(
            embeddings,
            positions,
            attention_mask,
            checkpoint_enabled=self.gradient_checkpointing,
            method="encode_embeddings",
        ).astype(self.compute_dtype)
