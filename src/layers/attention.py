import jax
import jax.numpy as jnp

from jax.nn import initializers
from typing import Dict
from config import TransformerConfig
from utils import softmax_3d
from layers.layer import Layer
from jax import lax, random
from jax._src.random import KeyArray


class Attention(Layer):
    def __init__(self, config: TransformerConfig, name: str) -> None:
        super().__init__()
        self.name = name
        self.key_query_value_shape = (
            config.nb_heads,
            config.key_query_dim,
            config.embedding_dim,
        )
        self.scale_factor = jnp.sqrt(config.key_query_dim)

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        _, *subkeys = random.split(key, 5)
        initializer = initializers.glorot_normal()
        return {
            "query_weights": initializer(
                key=subkeys[0], shape=self.key_query_value_shape, dtype=jnp.float32
            ),
            "key_weights": initializer(
                key=subkeys[1], shape=self.key_query_value_shape, dtype=jnp.float32
            ),
            "value_weights": initializer(
                key=subkeys[2], shape=self.key_query_value_shape, dtype=jnp.float32
            ),
            "output_weights": initializer(
                key=subkeys[3],
                shape=(
                    self.key_query_value_shape[2],
                    self.key_query_value_shape[0] * self.key_query_value_shape[1],
                ),
                dtype=jnp.float32,
            ),
        }

    def breakpoint_if_nonfinite(self, x: jnp.ndarray):
        is_finite = jnp.isnan(x).all()

        def true_fn(x):
            pass

        def false_fn(x):
            jax.debug.breakpoint()

        lax.cond(is_finite, false_fn, true_fn, x)

    def forward_simple(
        self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray
    ) -> jnp.ndarray:
        queries = jnp.matmul(weights["query_weights"], x)
        keys = jnp.matmul(weights["key_weights"], x)
        dots = jnp.matmul(keys.mT, queries) / self.scale_factor
        hm = jnp.triu(jnp.ones_like(dots))
        del queries
        del keys

        masked = jnp.where(hm, dots, float("-inf"))
        # self.breakpoint_if_nonfinite(masked)
        softmaxed = softmax_3d(masked)
        del dots
        del masked

        values = jnp.matmul(weights["value_weights"], x)
        summed = jnp.matmul(values, softmaxed)
        concat = jnp.concatenate(summed, axis=0)
        return jnp.dot(weights["output_weights"], concat)

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(self.forward_simple, in_axes=[None, 0])
        return batch_f(weights, x)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
