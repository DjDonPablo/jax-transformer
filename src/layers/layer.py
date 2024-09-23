from typing import Dict
from jax._src.random import KeyArray
import jax.numpy as jnp

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = "layer"

    @abstractmethod
    def forward(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def init_weights(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, weights: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
