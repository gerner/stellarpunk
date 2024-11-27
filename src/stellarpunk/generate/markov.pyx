from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string

import uroman

cdef extern from "markov.hpp":
    cdef cppclass MarkovModel5:
        void train_from_file(string filename)
        string generate(uint32_t seed)
        void save_to_file(string filename)
        void load_from_file(string filename)

romanizer=None

cdef class MarkovModel:
    cdef MarkovModel5 _model
    cdef bool _romanize
    cdef bool _titleize
    cdef bool _roman_numerals

    def __init__(self, romanize=False, titleize=True, roman_numerals=False):
        self._romanize=romanize
        self._titleize=titleize
        self._roman_numerals=roman_numerals

    def postprocess(self, example:str) -> str:
        if self._romanize:
            global romanizer
            assert romanizer
            example = romanizer.romanize_string(example)
        if self._titleize:
            example = example.title()
        if self._roman_numerals:
            parts = example.split()
            if parts[-1].lower() in ("i", "ii", "iii", "iv", "v", "vi" ,"vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"):
                parts[-1] = parts[-1].upper()
            example = " ".join(parts)
        return example


    def train(self, filename):
        if self._romanize:
            global romanizer
            if romanizer is None:
                romanizer = uroman.Uroman()

        self._model.train_from_file(filename.encode("utf-8"));

    def generate(self, random):
        return self.postprocess(self._model.generate(random.integers(2**32)).decode('utf-8'))

    def save(self, filename):
        self._model.save_to_file(filename.encode("utf-8"))

    def load(self, filename):
        self._model.load_from_file(filename.encode("utf-8"))

