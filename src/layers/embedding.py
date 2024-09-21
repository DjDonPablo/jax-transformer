from functools import partial
import jax.numpy as jnp

from config import TransformerConfig
from layers.layer import Layer
from jax import jit, random
from jax._src.random import KeyArray


class Embedding(Layer):
    def __init__(self, config: TransformerConfig, key: KeyArray) -> None:
        super().__init__()
        self.shape = (config.vocab_size, config.embedding_dim)
        self.weights = random.normal(key, self.shape, jnp.float32) * jnp.sqrt(
            1 / config.embedding_dim
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.weights[x].T

    def __str__(self) -> str:
        return f"Embedding<shape={self.shape}>"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)
