from functools import partial
import jax.numpy as jnp

from jax import jit
from abc import ABC, abstractmethod


class Layer(ABC):
    weights: jnp.ndarray

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
