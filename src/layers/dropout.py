import jax
import jax.numpy as jnp

from typing import Dict
from config import TransformerConfig
from layers.layer import Layer
from jax import random
from jax._src.random import KeyArray


class Dropout(Layer):
    def __init__(self, config: TransformerConfig, key: KeyArray, name: str) -> None:
        super().__init__()
        self.name = name
        self.key = key
        self.p = config.dropout_rate
        self.shape = (config.embedding_dim, config.context_size)
        self.training = config.training

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        return {}

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        if not self.training:
            return x
        self.key, k = jax.random.split(self.key)
        return random.bernoulli(k, self.p, self.shape) * x

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
