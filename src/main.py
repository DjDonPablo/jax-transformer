import jax.numpy as jnp

from jax import random
from config import TransformerConfig
from layers.embedding import Embedding
from layers.unembedding import Unembedding


if __name__ == "__main__":
    config = TransformerConfig(vocab_size=5, embedding_dim=10)
    key = random.key(42)
    embedding = Embedding(config=config, key=key)
    unembedding = Unembedding(config=config, key=key)

    # print(embedding)
    # print(embedding.weights)
    # print(embedding.forward(jnp.array(range(10)))[0][0].dtype)
    # print(unembedding.weights)
    # print(unembedding(jnp.array(range(10))))
