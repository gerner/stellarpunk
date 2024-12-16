import io
import json
import uuid
import struct
from typing import Union

import numpy as np
import numpy.typing as npt
import msgpack # type: ignore

from stellarpunk.serialization import serialize_econ_sim

def size_to_bytes(x:int) -> bytes:
    return x.to_bytes(4, byteorder="big", signed=False)

def size_to_f(x:int, f:io.IOBase) -> int:
    return f.write(size_to_bytes(x))

def size_from_f(f:io.IOBase) -> int:
    return int.from_bytes(f.read(4))

def float_to_f(value:float, f:io.IOBase) -> int:
    return f.write(struct.pack(">f", value))

def float_from_f(f:io.IOBase) -> float:
    return struct.unpack(">f", f.read(4))[0]

def int_to_f(x:int, f:io.IOBase, blen:int=4) -> int:
    return f.write(x.to_bytes(blen, byteorder="big", signed=False))

def int_from_f(f:io.IOBase, blen:int=4) -> int:
    return int.from_bytes(f.read(blen), byteorder="big", signed=False)

def to_len_pre_f(s:str, f:io.IOBase, blen:int=2) -> int:
    b = s.encode("utf8")
    prefix = len(b).to_bytes(blen)
    i = f.write(prefix)
    i += f.write(b)
    return i

def from_len_pre_f(f:io.IOBase, blen:int=2) -> str:
    prefix = f.read(blen)
    l = int.from_bytes(prefix)
    b = f.read(l)
    return b.decode("utf8")

def random_state_to_f(r:np.random.Generator, f:io.IOBase) -> int:
    # we assume this is a PCG64
    state = r.bit_generator.state
    s = json.dumps(state)
    return to_len_pre_f(s, f)

def random_state_from_f(f:io.IOBase) -> np.random.Generator:
    s = from_len_pre_f(f)
    state = json.loads(s)
    r = np.random.default_rng()
    r.bit_generator.state = state
    return r

def uuid_to_f(eid:uuid.UUID, f:io.IOBase) -> int:
    return f.write(eid.bytes)

def uuid_from_f(f:io.IOBase) -> uuid.UUID:
    return uuid.UUID(bytes=f.read(16))

def matrix_to_f(matrix:Union[npt.NDArray[np.float64], npt.NDArray[np.int64]], f:io.IOBase) -> int:
    ret = msgpack.packb(matrix, default = serialize_econ_sim.encode_matrix)
    size_to_f(len(ret), f)
    return f.write(ret)

def matrix_from_f(f:io.IOBase) -> npt.NDArray:
    count = size_from_f(f)
    matrix_bytes = f.read(count)
    return msgpack.unpackb(matrix_bytes, object_hook=serialize_econ_sim.decode_matrix)
