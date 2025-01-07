import io
import json
import uuid
import struct
from collections.abc import Collection, Mapping
from typing import Union, Callable, Optional, Union

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

def int_to_f(x:int, f:io.IOBase, blen:int=4, signed:bool=False) -> int:
    return f.write(x.to_bytes(blen, byteorder="big", signed=signed))

def int_from_f(f:io.IOBase, blen:int=4, signed:bool=False) -> int:
    return int.from_bytes(f.read(blen), byteorder="big", signed=signed)

def bytes_to_f(b:bytes, f:io.IOBase, blen:int=4) -> int:
    prefix = len(b).to_bytes(blen)
    i = f.write(prefix)
    i += f.write(b)
    return i

def bytes_from_f(f:io.IOBase, blen:int=4) -> bytes:
    prefix = f.read(blen)
    l = int.from_bytes(prefix)
    b = f.read(l)
    return b

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

def bool_to_f(b:bool, f:io.IOBase) -> int:
    return int_to_f(1 if b else 0, f, blen=1)

def bool_from_f(f:io.IOBase) -> bool:
    return int_from_f(f, blen=1) == 1

def primitive_to_f(x:Union[int,float,str,bool], f:io.IOBase, slen:int=2, ilen:int=4) -> int:
    bytes_written = 0
    if isinstance(x, int):
        bytes_written += f.write(b'i')
        bytes_written += int_to_f(x, f, blen=ilen)
    elif isinstance(x, float):
        bytes_written += f.write(b'f')
        bytes_written += float_to_f(x, f)
    elif isinstance(x, str):
        bytes_written += f.write(b's')
        bytes_written += to_len_pre_f(x, f, blen=slen)
    elif isinstance(x, bool):
        bytes_written += f.write(b'b')
        bytes_written += int_to_f(1 if x else 0, f, blen=1)
    else:
        raise ValueError(f'x must be int,float,str,bool. {x=}')
    return bytes_written

def primitive_from_f(f:io.IOBase, slen:int=2, ilen:int=4) -> Union[int,float,str,bool]:
    type_code = f.read(1)
    if type_code == b'i':
        return int_from_f(f, blen=ilen)
    elif type_code == b'f':
        return float_from_f(f)
    elif type_code == b's':
        return from_len_pre_f(f, blen=slen)
    elif type_code == b'b':
        return int_from_f(f, blen=1) == 1
    else:
        raise ValueError(f'got unexpected type {type_code=}')

def ints_to_f(seq:Collection[int], f:io.IOBase, blen:int=4, signed:bool=False) -> int:
    bytes_written = 0
    bytes_written += size_to_f(len(seq), f)
    for x in seq:
        bytes_written += int_to_f(x, f, blen=blen, signed=signed)
    return bytes_written

def ints_from_f(f:io.IOBase, blen:int=4, signed:bool=False) -> list[int]:
    count = size_from_f(f)
    seq = []
    for i in range(count):
        x = int_from_f(f, blen=blen, signed=signed)
        seq.append(x)
    return seq

def floats_to_f(seq:Collection[float], f:io.IOBase) -> int:
    bytes_written = 0
    bytes_written += size_to_f(len(seq), f)
    for x in seq:
        bytes_written += float_to_f(x, f)
    return bytes_written

def floats_from_f(f:io.IOBase) -> Collection[float]:
    count = size_from_f(f)
    seq = []
    for i in range(count):
        x = float_from_f(f)
        seq.append(x)
    return seq

def float_pair_to_f(a:Union[tuple[float, float], npt.NDArray[np.float64]], f:io.IOBase) -> int:
    bytes_written = 0
    bytes_written += float_to_f(a[0], f)
    bytes_written += float_to_f(a[1], f)
    return bytes_written

def float_pair_from_f(f:io.IOBase) -> npt.NDArray[np.float64]:
    x = float_from_f(f)
    y = float_from_f(f)
    return np.array((x, y))

def fancy_dict_to_f[K, V](d:Mapping[K, V], f:io.IOBase, s_k:Callable[[K, io.IOBase], int], s_v:Callable[[V, io.IOBase], int]) -> int:
    bytes_written = 0
    bytes_written += size_to_f(len(d), f)
    for k,v in d.items():
        bytes_written += s_k(k, f)
        bytes_written += s_v(v, f)
    return bytes_written

def fancy_dict_from_f[K, V](f:io.IOBase, s_k:Callable[[io.IOBase], K], s_v:Callable[[io.IOBase], V]) -> dict[K, V]:
    ret:dict[K, V] = {}
    count = size_from_f(f)
    for i in range(count):
        k = s_k(f)
        v = s_v(f)
        ret[k] = v
    return ret

def uuids_to_f(uuids:Collection[uuid.UUID], f:io.IOBase) -> int:
    bytes_written = 0
    bytes_written += size_to_f(len(uuids), f)
    for u in uuids:
        bytes_written += uuid_to_f(u, f)
    return bytes_written

def uuids_from_f(f:io.IOBase) -> list[uuid.UUID]:
    count = size_from_f(f)
    seq = []
    for i in range(count):
        x = uuid_from_f(f)
        seq.append(x)
    return seq

def str_uuids_to_f(items:Collection[tuple[str, uuid.UUID]], f:io.IOBase) -> int:
    bytes_written = 0
    bytes_written += size_to_f(len(items), f)
    for s,u in items:
        bytes_written += to_len_pre_f(s, f)
        bytes_written += uuid_to_f(u, f)
    return bytes_written

def str_uuids_from_f(f:io.IOBase) -> list[tuple[str, uuid.UUID]]:
    count = size_from_f(f)
    seq = []
    for i in range(count):
        s = from_len_pre_f(f)
        u = uuid_from_f(f)
        seq.append((s,u))
    return seq

def debug_string_w(s:str, f:io.IOBase) -> int:
    #TODO: flag to turn this off
    i = to_len_pre_f(s, f)
    return i

def debug_string_r(s:str, f:io.IOBase) -> str:
    #TODO: flag to turn this off
    f_pos_start = f.tell()
    prefix = f.read(2)
    l = int.from_bytes(prefix)
    b = f.read(l)
    s_actual = b.decode("utf8")
    if s != s_actual:
        f_pos = f.tell()
        assert s == s_actual
    return s_actual

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

def optional_uuid_to_f(eid:Optional[uuid.UUID], f:io.IOBase) -> int:
    bytes_written = 0
    if eid:
        bytes_written += bool_to_f(True, f)
        bytes_written += uuid_to_f(eid, f)
    else:
        bytes_written += bool_to_f(False, f)
    return bytes_written

def optional_uuid_from_f(f:io.IOBase) -> Optional[uuid.UUID]:
    if bool_from_f(f):
        return uuid_from_f(f)
    else:
        return None

def matrix_to_f(matrix:Union[npt.NDArray[np.float64], npt.NDArray[np.int64]], f:io.IOBase) -> int:
    ret = msgpack.packb(matrix, default = serialize_econ_sim.encode_matrix)
    bytes_written = size_to_f(len(ret), f)
    bytes_written += f.write(ret)
    return bytes_written

def matrix_from_f(f:io.IOBase) -> npt.NDArray:
    count = size_from_f(f)
    matrix_bytes = f.read(count)
    #TODO: we have to copy here because this decode path makes an ndarray with
    # some weird flags set, including WRITEABLE=False. I just haven't taken the
    # time to investigate this too much and loading is fast enough
    return msgpack.unpackb(matrix_bytes, object_hook=serialize_econ_sim.decode_matrix).copy()
