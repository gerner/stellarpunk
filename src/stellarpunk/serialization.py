""" Tools to save and load data. """

import sys
import io
import typing
import enum
from typing import BinaryIO, Optional, Tuple, Sequence, Any, Callable

import numpy as np
import numpy.typing as npt
import msgpack # type: ignore
import pandas as pd # type: ignore
from scipy import sparse # type: ignore

from stellarpunk import core

# numpy support courtsey:
# https://github.com/lebedov/msgpack-numpy/blob/master/msgpack_numpy.py

ENCODER_SIG = Optional[Callable[[Any], Any]]
DECODER_SIG = Optional[Callable[[Any], Any]]

class STypes(enum.IntEnum):
    PRODUCTION_CHAIN = enum.auto()

SFIELDS = {}

SFIELDS[STypes.PRODUCTION_CHAIN] = core.ProductionChain().__dict__.keys()

def ndarray_to_bytes(obj:typing.Any) -> typing.Any:
    if obj.dtype == 'O':
        return obj.dumps()
    else:
        if sys.platform == 'darwin':
            return obj.tobytes()
        else:
            return obj.data if obj.flags['C_CONTIGUOUS'] else obj.tobytes()

def _unpack_dtype(dtype:typing.Any) -> np.dtype:
    """
    Unpack dtype descr, recursively unpacking nested structured dtypes.
    """

    if isinstance(dtype, (list, tuple)):
        # Unpack structured dtypes of the form: (name, type, *shape)
        dtype = [
            (subdtype[0], _unpack_dtype(subdtype[1])) + tuple(subdtype[2:])
            for subdtype in dtype
        ]
    return np.dtype(dtype)

def encode_matrix(obj:typing.Any, chain:ENCODER_SIG=None) -> typing.Any:
    if isinstance(obj, np.ndarray):
        kind = b''
        descr = obj.dtype.str
        return {
                b'nd': True,
                b'type': descr,
                b'kind': kind,
                b'shape': obj.shape,
                b'data': ndarray_to_bytes(obj)
        }
    else:
        return obj if chain is None else chain(obj)

def decode_matrix(obj:typing.Any, chain:DECODER_SIG=None) -> typing.Any:
    if b'nd' in obj:
        descr = obj[b'type']
        return np.ndarray(buffer=obj[b'data'],
                          dtype=_unpack_dtype(descr),
                          shape=obj[b'shape'])
    else:
        return obj if chain is None else chain(obj)

class TickMatrixWriter:
    def __init__(self, f:typing.BinaryIO) -> None:
        self.packer = msgpack.Packer(default=encode_matrix)
        self.f = f

    def write(self, tick:int, matrix:npt.NDArray[np.float64]) -> int:
        ret = self.packer.pack({
            "t": tick,
            "m": matrix
        })
        return self.f.write(ret)

class TickMatrixReader:
    def __init__(self, f:typing.BinaryIO) -> None:
        self.unpacker = msgpack.Unpacker(f, object_hook=decode_matrix)

    def read(self) -> typing.Optional[typing.Tuple[int, npt.NDArray[np.float64]]]:
        try:
            ret = self.unpacker.unpack()
            return (ret["t"], ret["m"])
        except msgpack.OutOfData:
            return None

def encode_production_chain(obj:typing.Any, chain:ENCODER_SIG=None) -> typing.Any:
    if isinstance(obj, core.ProductionChain):
        return {
            "_sp_t": STypes.PRODUCTION_CHAIN,
            "d": dict(map(lambda x: (x,obj.__dict__[x]), SFIELDS[STypes.PRODUCTION_CHAIN])),
        }
    else:
        return encode_matrix(obj, chain=chain)

def decode_production_chain(obj:typing.Any, chain:DECODER_SIG=None) -> typing.Any:
    if "_sp_t" in obj and obj["_sp_t"] == int(STypes.PRODUCTION_CHAIN):
        production_chain = core.ProductionChain()
        production_chain.__dict__.update(map(lambda x: (x,obj["d"][x]), SFIELDS[STypes.PRODUCTION_CHAIN]))
        return production_chain
    else:
        return decode_matrix(obj, chain=chain)

def save_production_chain(production_chain:core.ProductionChain) -> bytes:
    return msgpack.packb(production_chain, default=encode_production_chain)

def load_production_chain(packed:bytes) -> core.ProductionChain:
    return msgpack.unpackb(packed, object_hook=decode_production_chain)

@typing.no_type_check
def _df_from_spmatrix(data:Any, index:Any=None, columns:Optional[Sequence[Any]]=None, fill_values:Optional[Any]=None) -> pd.DataFrame:
    """ Taken from https://github.com/pandas-dev/pandas/blob/5c66e65d7b9fef47ccb585ce2fd0b3ea18dc82ea/pandas/core/arrays/sparse/accessor.py 

    modified to allow setting fill_values """

    from pandas._libs.sparse import IntIndex # type: ignore

    from pandas import DataFrame # type: ignore

    data = data.tocsc()
    index, columns = DataFrame.sparse._prep_index(data, index, columns)
    n_rows, n_columns = data.shape
    # We need to make sure indices are sorted, as we create
    # IntIndex with no input validation (i.e. check_integrity=False ).
    # Indices may already be sorted in scipy in which case this adds
    # a small overhead.
    data.sort_indices()
    indices = data.indices
    indptr = data.indptr
    array_data = data.data
    arrays = []

    if fill_values is None:
        fill_values = [0] * n_columns
    elif not isinstance(fill_values, Sequence):
        fill_values = [fill_values] * n_columns

    for i, fill_value in zip(range(n_columns), fill_values):
        dtype = pd.SparseDtype(array_data.dtype, fill_value)
        sl = slice(indptr[i], indptr[i + 1])
        idx = IntIndex(n_rows, indices[sl], check_integrity=False)
        arr = pd.arrays.SparseArray._simple_new(array_data[sl], idx, dtype)
        arrays.append(arr)
    return DataFrame._from_arrays(
        arrays, columns=columns, index=index, verify_integrity=False
    )

def read_tick_log_to_df(f:BinaryIO, index_name:Optional[str]=None, column_names:Optional[Sequence[str]]=None, fill_values:Optional[Any]=None, sparse_matrix:bool=False) -> pd.DataFrame:
    reader = TickMatrixReader(f)
    matrixes = []
    row_count = 0
    col_count = 0
    ticks = []
    while (ret := reader.read()) is not None:
        tick, m = ret
        if row_count > 0:
            rows = m.shape[0]
            if len(m.shape) == 1:
                cols = 1
            else:
                cols = m.shape[1]
            if (row_count, col_count) != (rows, cols):
                raise ValueError(f'expected each matrix to have same shape {(row_count, col_count)} vs {m.shape}')
        else:
            assert col_count == 0
            row_count = m.shape[0]
            if len(m.shape) == 1:
                col_count = 1
            else:
                col_count = m.shape[1]
        if sparse_matrix:
            if col_count == 1:
                matrixes.append(sparse.csc_array(m[:,np.newaxis]))
            else:
                matrixes.append(sparse.csc_array(m))
        else:
            if col_count == 1:
                matrixes.append(m[:,np.newaxis])
            else:
                matrixes.append(m)
        ticks.append(np.full((row_count,), tick))

    if sparse_matrix:
        df = _df_from_spmatrix(sparse.vstack(matrixes), columns=column_names, fill_values=fill_values)
    else:
        df = pd.DataFrame(np.vstack(matrixes), columns=column_names)

    df["tick"] = pd.Series(np.concatenate(ticks))
    df.index = pd.Series(np.tile(np.arange(row_count), len(matrixes)))
    if index_name is not None:
        df.index.set_names(index_name, inplace=True)
    df.set_index("tick", append=True, inplace=True)

    return df
