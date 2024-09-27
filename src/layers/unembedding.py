from typing import Dict
import jax
import jax.numpy as jnp

from config import TransformerConfig
from utils import softmax_2d
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


class Unembedding(Layer):
    def __init__(self, config: TransformerConfig, name: str) -> None:
        super().__init__()
        self.name = name
        self.shape = (config.vocab_size, config.embedding_dim)
        self.temp = config.temp

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        return {
            "weights": random.normal(key, self.shape, jnp.float32)
            * jnp.sqrt(1 / self.shape[1])
        }

    def forward_simple(
        self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray
    ) -> jnp.ndarray:
        return softmax_2d(jnp.matmul(weights["weights"], x), self.temp)

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(self.forward_simple, in_axes=[None, 0])
        return batch_f(weights, x)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
