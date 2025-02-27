from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython import parallel

import os

import uroman

cdef extern from "markov.hpp":
    cdef cppclass CMarkovModel nogil:
        bool train_from_file(string filename)
        string generate(uint32_t seed)
        bool save_to_file(string filename)

    cdef void load_many_models(vector[CMarkovModel*] models, vector[string] filenames)
    cdef CMarkovModel* create_markov_model(size_t n)
    cdef CMarkovModel* load_markov_model(string filename)

romanizer=None

def load_models(filenames):
    models = []
    cdef vector[CMarkovModel*] cmodels;
    cdef vector[string] cfilenames;
    for filename in filenames:
        m = MarkovModel()
        models.append(m)
        cmodels.push_back(m._model)
        cfilenames.push_back(filename.encode("utf-8"))

    load_many_models(cmodels, cfilenames);

    return models

cdef class MarkovModel:
    cdef CMarkovModel* _model
    cdef bool _romanize
    cdef bool _titleize
    cdef bool _roman_numerals

    def __init__(self, romanize=False, titleize=True, roman_numerals=False, empty=False):
        self._romanize=romanize
        self._titleize=titleize
        self._roman_numerals=roman_numerals

        if empty:
            self._model = create_markov_model(5)
        else:
            self._model = NULL

    def __dealloc__(self):
        if self._model != NULL:
            del self._model
            self._model = NULL

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


    def train(self, filename, n=5):
        if self._model != NULL:
            del self._model
        self._model = create_markov_model(n)
        if self._romanize:
            global romanizer
            if romanizer is None:
                romanizer = uroman.Uroman()

        if not self._model.train_from_file(filename.encode("utf-8")):
            raise ValueError(f'could not train from {filename}')
        return self

    def generate(self, random):
        assert self._model != NULL
        return self.postprocess(self._model.generate(random.integers(2**32)).decode('utf-8'))

    def save(self, filename):
        assert self._model != NULL
        if not self._model.save_to_file(filename.encode("utf-8")):
            raise ValueError(f'could not save to {filename}')
        return self

    def load(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f'cannot load model, file does not exist {filename}')
        if self._model != NULL:
            del self._model
        self._model = load_markov_model(filename.encode("utf-8"))
        if self._model == NULL:
            raise ValueError(f'could not load from {filename}')
        if self._romanize:
            global romanizer
            if romanizer is None:
                romanizer = uroman.Uroman()

        return self

