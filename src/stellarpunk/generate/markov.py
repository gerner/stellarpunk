import io
import collections
from collections.abc import MutableMapping, Sequence, Iterator
from typing import Optional

import numpy.typing as npt
import numpy as np

TOKEN_START = 0
TOKEN_END = 1

class MarkovModel:
    def __init__(self, n:int=3):
        self.n = n

        # mapping from token id (index in list) to token
        self._tokens:list[str] = []
        self._token_ids:MutableMapping[str, int] = {}

        self._probabilities:MutableMapping[Sequence[int], npt.NDArray[np.float64]] = collections.defaultdict(lambda: np.zeros(len(self._tokens)))

    def clear(self) -> None:
        # mapping from token id (index in list) to token
        self._tokens = []
        self._token_ids = {}

        self._probabilities = collections.defaultdict(lambda: np.zeros(len(self._tokens)))

    def _process_example(self, example:str, ngram_counts:MutableMapping[Sequence[int], MutableMapping[int, int]]) -> None:

        # prepare the ngram with the start token
        ngram:collections.deque[int] = collections.deque()
        for _ in range(self.n-1):
            ngram.append(TOKEN_START)

        for token in self.tokenize(example):
            token_id = self._token_ids.get(token)
            if token_id is None:
                token_id = len(self._tokens)
                self._tokens.append(token)
                self._token_ids[token] = token_id

            ngram_counts[tuple(ngram)][token_id] += 1
            ngram.popleft()
            ngram.append(token_id)

        # add a count for the end token
        ngram_counts[tuple(ngram)][TOKEN_END] += 1
        ngram.popleft()

    def _build_probabilities(self, counts:MutableMapping[int, int]) -> npt.NDArray[np.float64]:
        probabilities = np.zeros(len(self._tokens))
        for token_id, count in counts.items():
            probabilities[token_id] = count
        return probabilities / probabilities.sum()

    def save(self, output_stream:io.BufferedWriter) -> None:
        """ saves this markov model to given stream"""

        # save n
        # save token mappings
        # for each n-1 gram, save each next token probabilities
        pass

    def load(self, input_stream:io.BufferedReader) -> None:
        pass

    def tokenize(self, example:str) -> Iterator[str]:
        """ tokenizes the example.

        in this case we're just iterating over characters. """
        for c in example:
            yield c

    def preprocess(self, example:str) -> str:
        return example.strip().lower()

    def postprocess(self, example:str) -> str:
        parts = example.title().split()
        if parts[-1].lower() in ("i", "ii", "iii", "iv", "v", "vi" ,"vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"):
            parts[-1] = parts[-1].upper()
        return " ".join(parts)

    def train(self, training_stream:io.TextIOBase, n:Optional[int]=None) -> None:
        """ trains this markov model

        assume one example per line. """

        if n is not None:
            self.n = n

        ngram_counts:MutableMapping[Sequence[int], MutableMapping[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

        # token 0 is start
        self._tokens.append("")
        # token 1 is end
        self._tokens.append("")

        for example in training_stream:
            self._process_example(self.preprocess(example), ngram_counts)

        for ngram, counts in ngram_counts.items():
            probabilities = self._build_probabilities(counts)
            self._probabilities[ngram] = probabilities


    def generate(self, r:np.random.Generator) -> str:
        """ generates one example """

        # prepare the ngram with the start token
        ngram:collections.deque[int] = collections.deque()
        for _ in range(self.n-1):
            ngram.append(TOKEN_START)

        probabilities = self._probabilities[tuple(ngram)]
        token_id = r.choice(len(probabilities), p=probabilities)
        result:collections.deque[int] = collections.deque()
        while token_id != TOKEN_END:
            result.append(token_id)
            ngram.popleft()
            ngram.append(token_id)
            probabilities = self._probabilities[tuple(ngram)]
            token_id = r.choice(len(probabilities), p=probabilities)

        return self.postprocess("".join(self._tokens[x] for x in result))
