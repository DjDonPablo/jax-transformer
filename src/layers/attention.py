import jax.numpy as jnp

from jax.nn import initializers
from typing import Dict
from config import TransformerConfig
from utils import softmax
from layers.layer import Layer
from jax import random
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

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        queries = jnp.matmul(weights["query_weights"], x)
        keys = jnp.matmul(weights["key_weights"], x)
        dots = jnp.triu(jnp.matmul(keys.mT, queries) / self.scale_factor)
        del queries
        del keys

        masked = dots.at[dots == 0].set(float("-inf"))
        softmaxed = softmax(masked, 1)
        del masked

        values = jnp.matmul(weights["value_weights"], x)
        summed = jnp.matmul(values, softmaxed)
        concat = jnp.concatenate(summed, axis=0)
        return jnp.dot(weights["output_weights"], concat)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
