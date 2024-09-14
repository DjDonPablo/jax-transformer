import jax.numpy as jnp

from config import TransformerConfig
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


class Unembedding(Layer):
    def __init__(self, config: TransformerConfig, key: KeyArray) -> None:
        super().__init__()
        self.shape = (config.embedding_dim, config.vocab_size)
        self.weights = random.normal(key, self.shape, jnp.float32) * jnp.sqrt(
            1 / config.embedding_dim
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.weights)

    def __str__(self) -> str:
        return f"Unembedding<shape={self.shape}>"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)
