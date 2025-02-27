from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython import parallel

import os

import uroman

cdef extern from "markov.hpp":
    cdef cppclass MarkovModel5 nogil:
        bool train_from_file(string filename)
        string generate(uint32_t seed)
        bool save_to_file(string filename)
        bool load_from_file(string filename)

    cdef void load_many_models(vector[MarkovModel5*] models, vector[string] filenames)

romanizer=None

def load_models(filenames):
    models = []
    cdef vector[MarkovModel5*] cmodels;
    cdef vector[string] cfilenames;
    for filename in filenames:
        m = MarkovModel()
        models.append(m)
        cmodels.push_back(&(m._model))
        cfilenames.push_back(filename.encode("utf-8"))

    load_many_models(cmodels, cfilenames);

    return models

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

        if not self._model.train_from_file(filename.encode("utf-8")):
            raise ValueError(f'could not train from {filename}')
        return self

    def generate(self, random):
        return self.postprocess(self._model.generate(random.integers(2**32)).decode('utf-8'))

    def save(self, filename):
        if not self._model.save_to_file(filename.encode("utf-8")):
            raise ValueError(f'could not save to {filename}')
        return self

    def load(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f'cannot load model, file does not exist {filename}')
        if not self._model.load_from_file(filename.encode("utf-8")):
            raise ValueError(f'could not load from {filename}')
        if self._romanize:
            global romanizer
            if romanizer is None:
                romanizer = uroman.Uroman()

        return self

