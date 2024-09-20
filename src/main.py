import jax
import jax.numpy as jnp

from jax import random
from config import TransformerConfig
from layers.attention import Attention
from layers.embedding import Embedding
from layers.unembedding import Unembedding


if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=5, embedding_dim=6, key_query_dim=3, nb_heads=2
    )
    key = random.key(42)
    embedding = Embedding(config=config, key=key)
    attention = Attention(config=config, key=key)

    embeddings = embedding(jnp.array(range(4)))
    res = attention(embeddings)
