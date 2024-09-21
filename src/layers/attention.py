import jax.numpy as jnp

from config import TransformerConfig
from utils import softmax
from layers.layer import Layer
from jax import jit, random
from jax._src.random import KeyArray


class Attention(Layer):
    def __init__(self, config: TransformerConfig, key: KeyArray) -> None:
        super().__init__()
        self.key_query_value_shape = (
            config.nb_heads,
            config.key_query_dim,
            config.embedding_dim,
        )
        # self.query_weights = jnp.zeros(
        #     shape=self.key_query_value_shape, dtype=jnp.float32
        # )
        # self.key_weights = jnp.zeros(
        #     shape=self.key_query_value_shape, dtype=jnp.float32
        # )
        # self.value_weights = jnp.zeros(
        #     shape=self.key_query_value_shape, dtype=jnp.float32
        # )

        # self.output_weights = jnp.zeros(
        #     shape=(config.key_query_dim * config.nb_heads, config.embedding_dim),
        #     dtype=jnp.float32,
        # )

        self.query_weights = random.normal(
            key=key, shape=self.key_query_value_shape, dtype=jnp.float32
        )
        self.key_weights = random.normal(
            key=key, shape=self.key_query_value_shape, dtype=jnp.float32
        )
        self.value_weights = random.normal(
            key=key, shape=self.key_query_value_shape, dtype=jnp.float32
        )

        self.output_weights = random.normal(
            key=key,
            shape=(config.embedding_dim, config.key_query_dim * config.nb_heads),
            dtype=jnp.float32,
        )

        self.scale_factor = jnp.sqrt(config.key_query_dim)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        queries = jnp.matmul(self.query_weights, x)
        keys = jnp.matmul(self.key_weights, x)
        dots = jnp.triu(jnp.matmul(keys.mT, queries) / self.scale_factor)
        del queries
        del keys

        masked = dots.at[dots == 0].set(float("-inf"))
        softmaxed = softmax(masked, 1)
        del masked

        values = jnp.matmul(self.value_weights, x)
        summed = jnp.matmul(values, softmaxed)
        concat = jnp.concatenate(summed, axis=0)
        return jnp.dot(self.output_weights, concat) + x

    def __str__(self) -> str:
        return f"Attention<key_query_shape={self.key_query_value_shape}, output_shape={self.output_weights.shape}>"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)
