import io
import collections
from collections.abc import MutableMapping, Sequence, Iterator

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

        self._ngram_counts:MutableMapping[Sequence[int], MutableMapping[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        # counts used in denominator, so n-1 gram counts
        self._base_counts:MutableMapping[Sequence[int], int] = collections.defaultdict(int)

        self._probabilities:npt.NDArray[np.float64] = np.ndarray(0)

    def clear(self) -> None:
        # mapping from token id (index in list) to token
        self._tokens = []
        self._token_ids = {}

        self._ngram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        # counts used in denominator, so n-1 gram counts
        self._base_counts = collections.defaultdict(int)

    def _process_example(self, example:str) -> None:

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

            self._base_counts[tuple(ngram)] += 1
            self._ngram_counts[tuple(ngram)][token_id] += 1
            ngram.popleft()
            ngram.append(token_id)

        # add a count for the end token
        self._base_counts[tuple(ngram)] += 1
        self._ngram_counts[tuple(ngram)][TOKEN_END] += 1
        ngram.popleft()

    def _build_probabilities(self, ngram:collections.deque) -> npt.NDArray[np.float64]:
        probabilities = np.zeros(len(self._tokens))
        for token_id, count in self._ngram_counts[tuple(ngram)].items():
            probabilities[token_id] = count
        return probabilities / probabilities.sum()


    def tokenize(self, example:str) -> Iterator[str]:
        """ tokenizes the example.

        in this case we're just iterating over characters. """
        for c in example:
            yield c

    def train(self, training_stream:io.TextIOBase) -> None:
        """ trains this markov model

        assume one example per line. """

        self.clear()

        # token 0 is start
        self._tokens.append("")
        # token 1 is end
        self._tokens.append("")

        for example in training_stream:
            self._process_example(example.strip())

    def generate(self, r:np.random.Generator) -> str:
        """ generates one example """

        # prepare the ngram with the start token
        ngram:collections.deque[int] = collections.deque()
        for _ in range(self.n-1):
            ngram.append(TOKEN_START)

        probabilities = self._build_probabilities(ngram)
        token_id = r.choice(len(probabilities), p=probabilities)
        result:collections.deque[int] = collections.deque()
        while token_id != TOKEN_END:
            result.append(token_id)
            ngram.popleft()
            ngram.append(token_id)
            probabilities = self._build_probabilities(ngram)
            token_id = r.choice(len(probabilities), p=probabilities)

        return "".join(self._tokens[x] for x in result)
