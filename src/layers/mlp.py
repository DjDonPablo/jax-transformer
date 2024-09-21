import jax
import jax.numpy as jnp

from config import TransformerConfig
from utils import softmax
from layers.layer import Layer
from jax import jit, random
from jax._src.random import KeyArray


class MLP(Layer):
    def __init__(self, config: TransformerConfig, key: KeyArray) -> None:
        super().__init__()
        self.linear1 = random.normal(
            key=key, shape=(4 * config.embedding_dim, config.embedding_dim), dtype=jnp.float32
        )
        self.bias1 = jnp.zeros(4 * config.embedding_dim)

        self.linear2 = random.normal(
            key=key, shape=(config.embedding_dim, 4 * config.embedding_dim), dtype=jnp.float32
        )
        self.bias2 = jnp.zeros(config.embedding_dim)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(self.linear2, jax.nn.gelu(jnp.matmul(self.linear1, x) + self.bias1)) + self.bias2 + x

    def __str__(self) -> str:
        return f"MLP<linear1_shape={self.linear1.shape}, linear2_shape={self.linear2.shape}>"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)
