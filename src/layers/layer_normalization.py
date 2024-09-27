import jax
import jax.numpy as jnp

from typing import Dict
from jax._src.random import KeyArray
from config import TransformerConfig
from layers.layer import Layer


class LayerNormalization(Layer):
    def __init__(
        self, config: TransformerConfig, name: str, epsilon: float = 1e-5
    ) -> None:
        super().__init__()
        self.name = name
        self.embedding_dim = config.embedding_dim
        self.epsilon = epsilon

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        return {
            "bias": jnp.zeros(self.embedding_dim),
            "gains": jnp.ones(self.embedding_dim),
        }

    def forward_simple(
        self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray
    ) -> jnp.ndarray:
        x = ((x.T - jnp.mean(x)) / jnp.sqrt(jnp.var(x) + self.epsilon)) * weights[
            "gains"
        ] + weights["bias"]
        return x.T

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(self.forward_simple, in_axes=[None, 0])
        return batch_f(weights, x)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
