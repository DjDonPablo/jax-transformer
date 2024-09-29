import jax
import jax.numpy as jnp
import os
import time
import sys
import random

from typing import Dict
from tokenizer import Tokenizer
from utils import get_token_from_softmax
from jax.example_libraries import optimizers
from config import TransformerConfig
from transformer import Transformer


def train_model(checkpoints_path: str = ""):
    # init config, transformer, weights...
    config = TransformerConfig(warmup_steps=1000, batch_size=32)
    transformer = Transformer(config)
    if checkpoints_path == "":
        step = 1
        transformer.init_weights()
    else:
        nb_checkpoints = len(os.listdir(checkpoints_path)) 
        step = nb_checkpoints * 100 + 1
        transformer.weights_dict = jnp.load(f"{checkpoints_path}/checkpoint_{nb_checkpoints}.npy", allow_pickle=True).item()

    # init optimizer
    opt_init, opt_update, get_params = optimizers.adam(
        config.get_lr, config.beta1, config.beta2, config.epsilon
    )
    opt_state = opt_init(transformer.weights_dict)

    # main loop
    loss_history = []
    nb_batch = len(os.listdir("../data/batch/X"))
    while step < 3000:
        batch = step % nb_batch + 1
        start = time.time()
        print(f"Step {step} - ", end="", flush=True)
        X, y = (
            jnp.load(f"../data/batch/X/batch_{batch}.npy"),
            jnp.load(f"../data/batch/y/batch_{batch}.npy"),
        )
        
        net_params = get_params(opt_state)
        loss, grads = jax.value_and_grad(transformer.loss)(net_params, X, y)

        opt_state = opt_update(step, grads, opt_state)
        print(f"loss = {loss}, time = {round(time.time() - start, 2)}s")

        step += 1
        loss_history.append(loss)

        if step % 100 == 0:
            jnp.save(f"../data/checkpoints/checkpoint_{step // 100}.npy", get_params(opt_state))

    jnp.save("../data/loss_history.npy", jnp.array(loss_history))
    print(loss_history)


def inference(checkpoint_path: str, temp: float = 1.0, optional_start: str = ""):
    config = TransformerConfig(training=False, temp=temp)
    tokenizer = Tokenizer(config.vocab_size, "../data/vocab-valid.txt", "../data/gpt4-valid-formatted.txt")
    transformer = Transformer(config)
    transformer.weights_dict = jnp.load(checkpoint_path, allow_pickle=True).item()

    x = jnp.array([[jax.nn.one_hot(config.vocab_size - 2, config.vocab_size)]]) # init X with start token

    key = jax.random.key(random.randint(0, 1000000))

    i = 0
    while i < 512:
        softmaxed = transformer.forward(transformer.weights_dict, x)
        key, subkey = jax.random.split(key)
        token_index = get_token_from_softmax(softmaxed[0], config.top_k, subkey)
        if token_index == config.vocab_size - 1:
            break
        print(tokenizer.get_token_from_index(token_index), end="", flush=True)
        x = jnp.append(x, jnp.array([[jax.nn.one_hot(token_index, config.vocab_size)]]), axis=1)

    print()

# not the best way to do cli but my way
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Choose --train {checkpoints_path} or --inference [checkpoint_path] {temp} {optional_start}")
    elif sys.argv[1] == "--train":
        if len(sys.argv) == 2:
            train_model()
        else:
            train_model(sys.argv[2])
    elif sys.argv[1] == "--inference":
        if len(sys.argv) == 3:
            inference(sys.argv[2])
        elif len(sys.argv) == 4:
            inference(sys.argv[2], float(sys.argv[3]))
        else:
            inference(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    else:
        print("Choose --train {checkpoints_path} or --inference [checkpoint_path] {temp} {optional_start}")