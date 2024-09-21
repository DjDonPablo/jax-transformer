from jax import jit
import jax
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
