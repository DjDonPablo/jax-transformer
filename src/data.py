import jax
import jax.numpy as jnp

from typing import List
from config import TransformerConfig
from tokenizer import Tokenizer


def save_vocab(path: str, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)


def format_file(path: str, save_path: str):
    with open(path) as f:
        lines = f.readlines()
    lim = len(lines)
    i = 0
    while i < lim:
        if len(lines[i]) < 3:
            del lines[i]
            i -= 1
            lim -= 1
        i += 1

    with open(save_path, "w") as f:
        for line in lines:
            f.write(line)


def pad_truncate_y(x: List[int], context_size: int, vocab_size: int, epsilon: float):
    y = jax.nn.one_hot(x[1:], vocab_size)[:context_size]
    y = jnp.where(y, 1 - epsilon, (epsilon / (vocab_size - 1)))  # label smoothing

    y_padded = list(
        jnp.concatenate([y, jnp.zeros((context_size - len(y), vocab_size))]).T
    )

    return (x[:context_size] + ([0] * (context_size - len(x))), y_padded)


def create_dataset(
    config: TransformerConfig,
    tokenizer: Tokenizer,
    data_path: str,
):
    print("Opening file...")

    with open(data_path) as f:
        batch_x = []
        batch_y = []
        story = ""
        batch_index = 0

        for i, line in enumerate(f):
            if line.strip() != "<|endoftext|>":
                story += line.strip()
            else:
                encoded_story = tokenizer.encode(story)
                x, y = pad_truncate_y(
                    encoded_story,
                    config.context_size,
                    tokenizer.vocab_size_limit,
                    config.label_smoothing,
                )

                batch_x.append(x)
                batch_y.append(y)

                if len(batch_x) >= config.batch_size:
                    save_batch(batch_x, batch_y, batch_index)

                    batch_x = []
                    batch_y = []
                    batch_index += 1

                story = ""

        if batch_x:
            save_batch(batch_x, batch_y, batch_index)


def save_batch(batch_x, batch_y, batch_index: int):
    print(f"Saving batch {batch_index}...")
    jnp.save(f"../data/batch/X/batch_{batch_index}.npy", jnp.array(batch_x))
    jnp.save(f"../data/batch/y/batch_{batch_index}.npy", jnp.array(batch_y))
    print(f"Saving for batch {batch_index} done !")
