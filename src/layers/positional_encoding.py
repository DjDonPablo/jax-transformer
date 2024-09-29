import jax
import jax.numpy as jnp

from typing import Dict
from jax._src.random import KeyArray
from config import TransformerConfig
from layers.layer import Layer


@jax.jit
def forward_simple_positional_encoding(
    x: jnp.ndarray, positional_encoding: jnp.ndarray
) -> jnp.ndarray:
    return x + positional_encoding[: x.shape[1]].T


class PositionalEncoding(Layer):
    def __init__(self, config: TransformerConfig, name: str) -> None:
        super().__init__()
        self.name = name

        pos = jnp.expand_dims(jnp.arange(config.context_size), 1)
        i = jnp.arange(0, config.embedding_dim, 2)

        self.positional_encoding = jnp.zeros(
            (config.context_size, config.embedding_dim)
        )

        self.positional_encoding = self.positional_encoding.at[:, 0::2].set(
            jnp.sin(pos / (10000 ** (i / config.embedding_dim)))
        )

        self.positional_encoding = self.positional_encoding.at[:, 1::2].set(
            jnp.cos(pos / (10000 ** (i / config.embedding_dim)))
        )

    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        return {}

    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        batch_f = jax.vmap(forward_simple_positional_encoding, in_axes=[0, None])
        return batch_f(x, self.positional_encoding)

    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(weights, x)
