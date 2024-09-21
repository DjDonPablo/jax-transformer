import jax
import jax.numpy as jnp

from jax import random
from config import TransformerConfig
from layers.attention import Attention
from layers.embedding import Embedding
from layers.mlp import MLP
from layers.positional_encoding import PositionalEncoding
from layers.unembedding import Unembedding


if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=5, embedding_dim=6, key_query_dim=3, nb_heads=2, context_size=12
    )
    key = random.key(42)
    embedding = Embedding(config=config, key=key)
    positional_encoding = PositionalEncoding(config=config)

    attention = Attention(config=config, key=key)
    mlp = MLP(config=config, key=key)

    unembedding = Unembedding(config=config, key=key)
