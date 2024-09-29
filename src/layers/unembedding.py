from typing import Dict
import jax
import jax.numpy as jnp

from config import TransformerConfig
from utils import softmax_2d
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


@jax.jit
def forward_simple_unembedding(
    weights: Dict[str, jnp.ndarray], x: jnp.ndarray, temp: float
) -> jnp.ndarray:
    return softmax_2d(jnp.matmul(weights["weights"], x), temp)


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


    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(forward_simple_unembedding, in_axes=[None, 0, None])
        return batch_f(weights, x, self.temp)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
