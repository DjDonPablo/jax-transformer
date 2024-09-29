import jax
import jax.numpy as jnp

from typing import Dict
from jax._src.random import KeyArray
from config import TransformerConfig
from layers.layer import Layer


@jax.jit
def forward_simple_layer_normalization(
        weights: Dict[str, jnp.ndarray], x: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
    x = ((x.T - jnp.mean(x)) / jnp.sqrt(jnp.var(x) + epsilon)) * weights[
            "gains"
    ] + weights["bias"]
    return x.T


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

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(forward_simple_layer_normalization, in_axes=[None, 0, None])
        return batch_f(weights, x, self.epsilon)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
