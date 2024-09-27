import jax
import jax.numpy as jnp

from jax.example_libraries import optimizers
from config import TransformerConfig
from transformer import Transformer

jax.config.update("jax_debug_nans", True)


def train_model():
    config = TransformerConfig(warmup_steps=1000)
    transformer = Transformer(config)
    if not config.use_existing_weights:
        step = 1
        transformer.init_weights()
        transformer.init_batched_functions()
    else:
        # load steps from existing directory
        return

    opt_init, opt_update, get_params = optimizers.adam(
        config.get_lr, config.beta1, config.beta2, config.epsilon
    )
    opt_state = opt_init(transformer.weights_dict)

    loss_history = []
    while step < 500:
        batch = step % 27
        print(f"Step {step} - ", end="", flush=True)
        X, y = (
            jnp.load(f"../data/batch/X/batch_{batch}.npy"),
            jnp.load(f"../data/batch/y/batch_{batch}.npy"),
        )
        X = jax.nn.one_hot(X, config.vocab_size)
        net_params = get_params(opt_state)
        loss, grads = jax.value_and_grad(transformer.loss)(net_params, X[:16], y[:16])
        print(f"loss = {loss}")

        opt_state = opt_update(step, grads, opt_state)

        step += 1
        loss_history.append(loss)

    print(loss_history)
    # how to save weights
    # checkpoints


if __name__ == "__main__":
    train_model()
    # batch = 0
    # X, y = (
    #     jnp.load(f"../data/batch/X/batch_{batch}.npy"),
    #     jnp.load(f"../data/batch/y/batch_{batch}.npy"),
    # )
    # print(X)
    # print(jax.nn.one_hot(X, 10256))
    # config = TransformerConfig()
    # transformer = Transformer(config)
    # transformer.init_weights()
    # transformer.init_batched_functions()
    # config = TransformerConfig()
    # tokenizer = Tokenizer(
    #     config.vocab_size, "../data/vocab-valid.txt", "../data/gpt4-valid-formatted.txt"
    # )
    # # tokenizer.bpe()
    # data.create_dataset(config, tokenizer, "../data/small_data.txt")

    # data.create_dataset("formatted.txt", "../data/dataset")
    # config = TransformerConfig(embedding_dim=6, context_size=5, vocab_size=6)
    # transformer = Transformer(config)
    # transformer.init_weights()
    # transformer.init_batched_functions()
    # start = time.time()
    # X = jnp.array([[1, 2, 3, 0, 0]])
    # y = jnp.array(
    #     [
    #         [
    #             [0, 0, 1, 0, 0],
    #             [0, 0, 0, 0, 0],
    #             [1, 0, 0, 0, 0],
    #             [0, 1, 0, 0, 0],
    #             [0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0],
    #         ]
    #     ]
    # )

    # print(transformer.loss(transformer.weights_dict, X, y))
    # print(time.time() - start)
