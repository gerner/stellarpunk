""" Tools to save and load data. """

import sys
import io
import typing
import enum

import numpy as np
import numpy.typing as npt
import msgpack # type: ignore

from stellarpunk import core

# numpy support courtsey:
# https://github.com/lebedov/msgpack-numpy/blob/master/msgpack_numpy.py

ENCODER_SIG = typing.Optional[typing.Callable[[typing.Any], typing.Any]]
DECODER_SIG = typing.Optional[typing.Callable[[typing.Any], typing.Any]]

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

