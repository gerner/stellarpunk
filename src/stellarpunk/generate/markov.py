import io
import collections
import json
from collections.abc import MutableMapping, Sequence, Iterator
from typing import Optional

import numpy.typing as npt
import numpy as np
import uroman # type:ignore

TOKEN_START = 0
TOKEN_END = 1



class MarkovModel:
    romanizer=uroman.Uroman()
    def __init__(self, n:int=3, romanize:bool=True, titleize:bool=True, roman_numerals:bool=False):
        self.n = n

        # mapping from token id (index in list) to token
        self._tokens:list[str] = []

        self._probabilities:MutableMapping[Sequence[int], tuple[Sequence[int], npt.NDArray[np.float64]]] = collections.defaultdict(lambda: ([], np.array([])))

        # post processing options
        # romanize generated samples to latin alphabet e.g. translating from
        # arabic characters
        self.romanize = romanize
        # title case words
        self.titleize = titleize
        # recognize and capitalize roman numerals as the final word
        self.roman_numerals = roman_numerals

    def clear(self) -> None:
        # mapping from token id (index in list) to token
        self._tokens = []
        # token 0 is start
        self._tokens.append("")
        # token 1 is end
        self._tokens.append("")

        self._probabilities = collections.defaultdict(lambda: ([], np.array([])))

    def _process_example(self, example:str, ngram_counts:MutableMapping[Sequence[int], MutableMapping[int, int]], token_ids:MutableMapping[str, int]) -> None:

        # skip empty string
        if not example:
            return

        # prepare the ngram with the start token
        ngram:collections.deque[int] = collections.deque()
        for _ in range(self.n-1):
            ngram.append(TOKEN_START)

        for token in self.tokenize(example):
            token_id = token_ids.get(token)
            if token_id is None:
                token_id = len(self._tokens)
                self._tokens.append(token)
                token_ids[token] = token_id

            ngram_counts[tuple(ngram)][token_id] += 1
            ngram.popleft()
            ngram.append(token_id)

        # add a count for the end token
        ngram_counts[tuple(ngram)][TOKEN_END] += 1
        ngram.popleft()

    def _build_probabilities(self, counts:MutableMapping[int, int]) -> tuple[Sequence[int], npt.NDArray[np.float64]]:
        probabilities = np.array(list(counts.values()))
        return (list(counts.keys()), probabilities / probabilities.sum())

    def save(self, output_stream:io.BufferedWriter) -> None:
        """ saves this markov model to given stream"""

        # save n
        # save token mappings
        # for each n-1 gram, save each next token probabilities
        n_minus_one_grams:list[Sequence[int]] = []
        probs:list[tuple[Sequence[int], Sequence[float]]] = []
        for key, (token_ids, probs_ndarray) in self._probabilities.items():
            n_minus_one_grams.append(key)
            probs.append((token_ids, list(probs_ndarray)))
        json.dump({
                "n": self.n,

                "romanize": self.romanize,
                "titleize": self.titleize,
                "roman_numerals": self.roman_numerals,
                "n_minus_one_grams":n_minus_one_grams,
                "probs": probs,
                "tokens": self._tokens,
            },
            io.TextIOWrapper(output_stream)
        )

    def load(self, input_stream:io.BufferedReader) -> None:
        self.clear()
        obj = json.load(input_stream)
        for nmo_gram, (token_ids, probs) in zip(obj["n_minus_one_grams"], obj["probs"]):
            self._probabilities[tuple(nmo_gram)] = (token_ids, np.array(probs))
        self._tokens = obj["tokens"]
        self.n = obj["n"]
        self.romanize = obj["romanize"]
        self.titleize = obj["titleize"]
        self.roman_numerals = obj["roman_numerals"]

    def tokenize(self, example:str) -> Iterator[str]:
        """ tokenizes the example.

        in this case we're just iterating over characters. """
        for c in example:
            yield c

    def preprocess(self, example:str) -> str:
        return example.strip().lower()

    def postprocess(self, example:str) -> str:
        if self.romanize:
            example = MarkovModel.romanizer.romanize_string(example)
        if self.titleize:
            example = example.title()
        if self.roman_numerals:
            parts = example.split()
            if parts[-1].lower() in ("i", "ii", "iii", "iv", "v", "vi" ,"vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"):
                parts[-1] = parts[-1].upper()
            example = " ".join(parts)
        return example

    def train(self, training_stream:io.TextIOBase, n:Optional[int]=None) -> None:
        """ trains this markov model

        assume one example per line. """

        self.clear()

        if n is not None:
            self.n = n

        ngram_counts:MutableMapping[Sequence[int], MutableMapping[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        token_ids:MutableMapping[str, int] = {}

        for example in training_stream:
            self._process_example(self.preprocess(example), ngram_counts, token_ids)

        for ngram, counts in ngram_counts.items():
            probabilities = self._build_probabilities(counts)
            self._probabilities[ngram] = probabilities


    def generate(self, r:np.random.Generator) -> str:
        """ generates one example """

        # prepare the ngram with the start token
        ngram:collections.deque[int] = collections.deque()
        for _ in range(self.n-1):
            ngram.append(TOKEN_START)

        (token_ids, probabilities) = self._probabilities[tuple(ngram)]
        token_id = r.choice(token_ids, p=probabilities)
        result:collections.deque[int] = collections.deque()
        while token_id != TOKEN_END:
            result.append(token_id)
            ngram.popleft()
            ngram.append(token_id)
            token_ids, probabilities = self._probabilities[tuple(ngram)]
            token_id = r.choice(token_ids, p=probabilities)

        return self.postprocess("".join(self._tokens[x] for x in result))
