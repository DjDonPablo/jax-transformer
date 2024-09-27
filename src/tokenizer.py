import regex as re
import ast
import os

from collections import defaultdict
from utils import load_data, save_data
from typing import List, Tuple


class Tokenizer:
    def __init__(self, vocab_size_limit: int, vocab_path: str, data_path: str):
        self.vocab_size_limit = vocab_size_limit
        self.pre_regex = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.num_regex = re.compile(r"(\d+)")

        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.vocab = [ast.literal_eval(line) for line in f if line.strip()][0]
        else:
            self.vocab = [i for i in range(256)]
        self.vocab_path = vocab_path
        self.data_path = data_path

    def word_frequencies_from_corpus(self) -> List[Tuple]:
        corpus = load_data(self.data_path)
        words = self.pre_regex.findall(corpus)

        unique_words = set(words)
        word_frequencies = [
            (list(key.encode("utf-8")), words.count(key)) for key in unique_words
        ]
        return word_frequencies  # [([l1, l2, l3, ...], nb_occurence), ...]

    def find_most_occuring_pair(self, word_frequencies: List[Tuple[List, int]]):
        pair_occurences = defaultdict(lambda: 0)
        for e in word_frequencies:
            len_word = len(e[0])
            for i in range(len_word - 1):
                pair_occurences[(e[0][i], e[0][i + 1])] += e[1]

        return max(pair_occurences.items(), key=lambda x: x[1])

    def update_word_frequencies(
        self, word_frequencies: List[Tuple[List, int]], pair: Tuple
    ):
        for e in word_frequencies:
            len_word = len(e[0])
            i = 0
            while i < len_word - 1:
                if (e[0][i], e[0][i + 1]) == pair:
                    del e[0][i + 1]
                    e[0][i] = pair
                    len_word -= 1
                i += 1

    def bpe(self):
        word_frequencies = self.word_frequencies_from_corpus()

        while len(self.vocab) < self.vocab_size_limit - 2:
            most_occuring_pair = self.find_most_occuring_pair(word_frequencies)
            if most_occuring_pair[1] == 1:  # ----- FOR TESTING PURPOSE ------
                break

            self.update_word_frequencies(word_frequencies, most_occuring_pair[0])

            self.vocab.append(most_occuring_pair[0])

        self.vocab.append(len(self.vocab))  # bos token
        self.vocab.append(len(self.vocab))  # eof token

        save_data(self.vocab_path, str(self.vocab))

    def encode(self, s: str) -> List[int]:
        len_voc = len(self.vocab)
        s_bytes = list(s.encode("utf-8"))

        for i in range(256, len_voc):
            len_s_bytes = len(s_bytes)
            j = 0
            while j < len_s_bytes - 1:
                if (s_bytes[j], s_bytes[j + 1]) == self.vocab[i]:
                    del s_bytes[j + 1]
                    s_bytes[j] = self.vocab[i]
                    len_s_bytes -= 1
                else:
                    j += 1

        # added eot token at beginning and end of string
        return (
            [self.vocab[-2]]
            + [self.vocab.index(token) for token in s_bytes]
            + [self.vocab[-1]]
        )

    def decode(self, i_tokens: List[int]):
        tokens = [self.vocab[i_token] for i_token in i_tokens]

        # s = []
        b = bytearray()
        for token in tokens:
            b.extend([int(byte) for byte in self.num_regex.findall(str(token))])
            # s.append(
            #     "".join(chr(int(byte)) for byte in self.num_regex.findall(str(token)))
            # )
        return b.decode("utf-8")

    def get_token_from_index(self, i_token: int):
        return "".join(
            chr(int(byte)) for byte in self.num_regex.findall(str(self.vocab[i_token]))
        )

    def get_vocab_pretty(self):
        return self.decode([i for i in range(len(self.vocab))])
