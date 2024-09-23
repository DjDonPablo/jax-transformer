import jax

from jax._src.random import KeyArray
import jax.numpy as jnp


def load_data(path: str) -> str:
    with open(path) as file:
        return file.read()


def save_data(path: str, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)


def softmax(x: jnp.ndarray, axis: int, temp: float = 1):
    x_max = jnp.max(x, axis, keepdims=True)
    unnormalized = jnp.exp((x - x_max) / temp)
    result = unnormalized / jnp.sum(unnormalized, axis, keepdims=True)
    return result


def get_token_from_softmax(softmaxed: jnp.ndarray, top_k: int, key: KeyArray):
    values, indices = jax.lax.top_k(softmaxed[:, -1], top_k)
    probs = values / values.sum()
    n = jax.random.uniform(key)
    return indices[jnp.argmax(n < jnp.cumsum(probs))]
