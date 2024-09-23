import jax.numpy as jnp

from jax.nn import initializers, gelu
from typing import Dict
from config import TransformerConfig
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


class MLP(Layer):
    def __init__(self, config: TransformerConfig, name: str) -> None:
        super().__init__()
        self.name = name
        self.embedding_dim = config.embedding_dim

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        _, *subkeys = random.split(key)
        initializer = initializers.glorot_normal()
        return {
            "linear1": initializer(
                key=subkeys[0],
                shape=(4 * self.embedding_dim, self.embedding_dim),
                dtype=jnp.float32,
            ),
            "bias1": jnp.zeros(4 * self.embedding_dim),
            "linear2": initializer(
                key=subkeys[1],
                shape=(self.embedding_dim, 4 * self.embedding_dim),
                dtype=jnp.float32,
            ),
            "bias2": jnp.zeros(self.embedding_dim),
        }

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.matmul(
                weights["linear2"],
                gelu(jnp.matmul(weights["linear1"], x) + weights["bias1"]),
            )
            + weights["bias2"]
        )

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
