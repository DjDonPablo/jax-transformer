import jax
import jax.numpy as jnp
from jax import grad

grad_tanh = grad(jnp.tanh)
print(grad_tanh(2.0))

print(grad(grad(jnp.tanh))(2.0))
print(grad(grad(grad(jnp.tanh)))(2.0))

f = lambda x: x**3 + 2 * x**2 - 3 * x + 1

dfdx = jax.grad(f)

d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)

print(dfdx(1.0))
print(d2fdx(1.0))
print(d3fdx(1.0))
print(d4fdx(1.0))

key = jax.random.key(0)


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)


# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Build a toy dataset.
inputs = jnp.array(
    [[0.52, 1.12, 0.77], [0.88, -1.08, 0.15], [0.52, 0.06, -1.30], [0.74, -2.49, 1.39]]
)
targets = jnp.array([True, True, False, True])


# Training loss is the negative log-likelihood of the training examples.
def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))


# Initialize random model coefficients
key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (3,))
b = jax.random.normal(b_key, ())

# Differentiate `loss` with respect to the first positional argument:
W_grad = grad(loss, argnums=0)(W, b)
print(f"{W_grad=}")

# Since argnums=0 is the default, this does the same thing:
W_grad = grad(loss)(W, b)
print(f"{W_grad=}")

# But you can choose different values too, and drop the keyword:
b_grad = grad(loss, 1)(W, b)
print(f"{b_grad=}")

# Including tuple values
W_grad, b_grad = grad(loss, (0, 1))(W, b)
print(f"{W_grad=}")
print(f"{b_grad=}")


def loss2(params_dict):
    preds = predict(params_dict["abc"]["W"], params_dict["abc"]["b"], inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))


print(grad(loss2)({"abc": {"W": W, "b": b}}))
