import jax.numpy as jnp

from jax import random
from config import TransformerConfig
from layers.attention import Attention
from layers.embedding import Embedding
from layers.layer_normalization import LayerNormalization
from layers.mlp import MLP
from layers.positional_encoding import PositionalEncoding
from layers.unembedding import Unembedding
import utils


if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=5,
        embedding_dim=5,
        key_query_dim=3,
        nb_heads=2,
        context_size=12,
        top_k=3,
    )
    key = random.key(42)
    embedding = Embedding(config=config, key=key)
    unembedding = Unembedding(config=config, key=key)
    res = unembedding(embedding(jnp.array(range(4))))
    print(res)
    utils.get_token_from_softmax(res, config.top_k, key)

    # positional_encoding = PositionalEncoding(config=config)
    # layer_normalization = LayerNormalization(config=config)

    # attention = Attention(config=config, key=key)
    # mlp = MLP(config=config, key=key)
