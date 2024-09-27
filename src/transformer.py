import jax
import jax.numpy as jnp

from typing import Callable, Dict, List
from config import TransformerConfig
from layers.attention import Attention
from layers.dropout import Dropout
from layers.embedding import Embedding
from layers.layer import Layer
from layers.layer_normalization import LayerNormalization
from layers.mlp import MLP
from layers.positional_encoding import PositionalEncoding
from layers.unembedding import Unembedding
from utils import cross_entropy_loss, softmax_2d


class Transformer:
    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.weights_dict = {}
        self.batched_forward: None | Callable = None
        self.batched_cel: None | Callable = None

        self.key = jax.random.key(69)
        self.key, *subkeys = jax.random.split(self.key, 1 + 1 + (config.nb_layers * 2))
        self.layers: List[Layer] = []
        self.embedding = Embedding(config=self.config, name="embedding")
        self.unembedding = Unembedding(config=config, name="unembedding")

        self.layers.append(Embedding(config=self.config, name="embedding"))
        self.layers.append(
            PositionalEncoding(config=config, name="positional_encoding")
        )
        self.layers.append(Dropout(config=config, key=subkeys[0], name="dropout_0"))

        subkeys_index = 1
        for layer in range(config.nb_layers):
            self.layers.append(Attention(config=config, name=f"attention_{layer}"))
            self.layers.append(
                Dropout(
                    config=config,
                    key=subkeys[subkeys_index],
                    name=f"dropout_{2 * layer + 1}",
                )
            )
            # residual connection
            self.layers.append(
                LayerNormalization(
                    config=config, name=f"layer_normalization_{2 * layer}"
                )
            )

            self.layers.append(MLP(config=config, name=f"mlp_{layer}"))
            self.layers.append(
                Dropout(
                    config=config,
                    key=subkeys[subkeys_index + 1],
                    name=f"dropout_{2 * (layer + 1)}",
                )
            )
            # residual connection
            self.layers.append(
                LayerNormalization(
                    config=config, name=f"layer_normalization_{2 * layer + 1}"
                )
            )
            subkeys_index += 2

        self.layers.append(Unembedding(config=config, name="unembedding"))

    def init_weights(self):
        self.key, *subkeys = jax.random.split(self.key, 1 + len(self.layers))
        for i, layer in enumerate(self.layers):
            self.weights_dict[layer.name] = layer.init_weights(subkeys[i])

    def init_batched_functions(self):
        self.batched_forward = jax.vmap(self.forward, in_axes=[None, 0])
        self.batched_cel = jax.vmap(cross_entropy_loss)

    def forward(self, weights_dict: Dict[str, Dict[str, jnp.ndarray]], x: jnp.ndarray):
        residual = x
        for layer in self.layers:
            if "attention" in layer.name or "mlp" in layer.name:
                residual = x
            x = layer(weights_dict[layer.name], x)
            if "dropout" in layer.name and layer.name[-1] != "0":
                x = x + residual  # dropout + residual connection

        return x

    def loss(
        self,
        weights_dict: Dict[str, Dict[str, jnp.ndarray]],
        X: jnp.ndarray,
        y: jnp.ndarray,
    ):
        return jnp.mean(cross_entropy_loss(self.forward(weights_dict, X), y))
