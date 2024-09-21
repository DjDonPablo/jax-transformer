import jax.numpy as jnp

from config import TransformerConfig
from layers.layer import Layer


class PositionalEncoding(Layer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
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

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.positional_encoding[: x.shape[1]].T

    def __str__(self) -> str:
        return "PositionalEncoding<>"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)
