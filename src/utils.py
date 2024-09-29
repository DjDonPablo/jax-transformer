import jax

from jax._src.random import KeyArray
import jax.numpy as jnp


def load_data(path: str) -> str:
    with open(path) as file:
        return file.read()


def save_data(path: str, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)


@jax.jit
def softmax_2d(x: jnp.ndarray, temp: float = 1):
    exp = jnp.exp(x / temp)
    return (exp.T / jnp.expand_dims(jnp.sum(exp, axis=0), 1)).T


@jax.jit
def softmax_3d(x: jnp.ndarray, temp: float = 1):
    exp = jnp.exp(x / temp)
    return (exp.mT / jnp.expand_dims(jnp.sum(exp, axis=1), 2)).mT


def get_token_from_softmax(softmaxed: jnp.ndarray, top_k: int, key: KeyArray):
    values, indices = jax.lax.top_k(softmaxed[:, -1], top_k)
    probs = values / values.sum()
    n = jax.random.uniform(key)
    return indices[jnp.argmax(n < jnp.cumsum(probs))]


@jax.jit
def cross_entropy_loss_simple(preds: jnp.ndarray, y: jnp.ndarray):
    return -jnp.sum(y * jnp.log(preds), axis=0).mean()

@jax.jit
def cross_entropy_loss(preds: jnp.ndarray, y: jnp.ndarray):
    batch_f = jax.vmap(cross_entropy_loss_simple)
    return batch_f(preds, y)
