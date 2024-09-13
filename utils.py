from jax import jit
import jax.numpy as jnp


def load_data(path: str) -> str:
    with open(path) as file:
        return file.read()


def save_data(path: str, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)


@jit
def softmax(x: jnp.ndarray, temp: float) -> jnp.ndarray:
    x = jnp.exp(x / temp)
    return x / jnp.mean(x)
