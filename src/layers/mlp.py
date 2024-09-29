import jax
import jax.numpy as jnp

from jax.nn import initializers, gelu
from typing import Dict
from config import TransformerConfig
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


@jax.jit
def forward_simple_mlp(
    weights: Dict[str, jnp.ndarray], x: jnp.ndarray
) -> jnp.ndarray:
    return (
        jnp.matmul(
            weights["linear2"],
            gelu(jnp.matmul(weights["linear1"], x) + weights["bias1"]),
        )
        + weights["bias2"]
    )

class MLP(Layer):
    def __init__(self, config: TransformerConfig, name: str) -> None:
        super().__init__()
        self.name = name
        self.embedding_dim = config.embedding_dim

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        _, *subkeys = random.split(key, 3)
        initializer = initializers.glorot_normal()
        return {
            "linear1": initializer(
                key=subkeys[0],
                shape=(4 * self.embedding_dim, self.embedding_dim),
                dtype=jnp.float32,
            ),
            "bias1": jnp.expand_dims(jnp.zeros(4 * self.embedding_dim), 1),
            "linear2": initializer(
                key=subkeys[1],
                shape=(self.embedding_dim, 4 * self.embedding_dim),
                dtype=jnp.float32,
            ),
            "bias2": jnp.expand_dims(jnp.zeros(self.embedding_dim), 1),
        }


    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(forward_simple_mlp, in_axes=[None, 0])
        return batch_f(weights, x)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
