import jax
import jax.numpy as jnp

from typing import Dict
from config import TransformerConfig
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


class Embedding(Layer):
    def __init__(self, config: TransformerConfig, name: str) -> None:
        super().__init__()
        self.name = name
        self.shape = (config.vocab_size, config.embedding_dim)

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        return {
            "weights": random.normal(key, self.shape, jnp.float32)
            * jnp.sqrt(1 / self.shape[1])
        }

    def forward_simple(
        self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.matmul(weights["weights"].T, x.T)  # weights["weights"][x].T

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(self.forward_simple, in_axes=[None, 0])
        return batch_f(weights, x)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
