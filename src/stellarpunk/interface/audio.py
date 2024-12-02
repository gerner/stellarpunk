""" Audio interface stuff """

import logging
import ctypes
import collections
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Deque

# TODO: figure out sdl distribution solution
import sdl2dll # type: ignore
import sdl2 # type: ignore
from sdl2 import sdlmixer as mixer
import numpy as np
import numpy.typing as npt

from stellarpunk import interface ,util

@dataclass
class Sample:
    sample_buffer: ctypes.Array[ctypes.c_ubyte]
    chunk: mixer.Mix_Chunk
    callback: Optional[Callable[[], Any]]

class Mixer(interface.AbstractMixer):

    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        # hang on to the ctypes function pointer so it doesn't get cleaned up
        self._channel_finished_callback = ctypes.CFUNCTYPE(
            None, ctypes.c_int
        )(self._channel_finished)

        self._sample_rate = 44100
        self.dtype = np.int16
        self.max_int = np.iinfo(np.int16).max

        self.channel_samples: Dict[int, Deque[Sample]] = collections.defaultdict(collections.deque)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __enter__(self) -> "Mixer":
        self.logger.info("intializing SDL")
        ret = sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)
        if ret < 0:
            err = mixer.Mix_GetError().decode("utf8")
            raise RuntimeError("Error initializing SDL: {0}".format(err))
        mixer.Mix_Init(0)

        # open at 44.1khz, 16-bit mono audio with a 1024 buffer for audio
        ret = mixer.Mix_OpenAudio(self.sample_rate, sdl2.AUDIO_S16SYS, 1, 1024)
        if ret < 0:
            err = mixer.Mix_GetError().decode("utf8")
            raise RuntimeError("Error initializing the mixer: {0}".format(err))

        # set a channel finished callback to signal when samples are done playing
        mixer.Mix_ChannelFinished(self._channel_finished_callback)

        return self

    def __exit__(self, type:Any, value:Any, traceback:Any) -> None:
        self.logger.info("shutting down SDL")
        mixer.Mix_Quit()
        sdl2.SDL_Quit()

    def _channel_finished(self, channel: int) -> None:
        sample = self.channel_samples[channel].popleft()
        # free the chunk
        mixer.Mix_FreeChunk(sample.chunk)
        # make callback if any
        if sample.callback:
            sample.callback()

    def play_sample(self, sample: npt.NDArray[np.float64], callback: Optional[Callable[[], Any]] = None, loops:int=0) -> int:
        arr = (np.clip(sample, -1, 1) * self.max_int).astype(self.dtype)

        # Cast the array into ctypes format for use with mixer
        arr_bytes = arr.tobytes()
        buflen = len(arr_bytes)
        c_buf = (ctypes.c_ubyte * buflen).from_buffer_copy(arr_bytes)

        # Convert the ctypes memory buffer into a mixer audio clip
        sample_chunk = mixer.Mix_QuickLoad_RAW(
            ctypes.cast(c_buf, ctypes.POINTER(ctypes.c_ubyte)), buflen
        )

        channel = mixer.Mix_GroupAvailable(-1)
        self.channel_samples[channel].append(Sample(c_buf, sample_chunk, callback))
        ret = mixer.Mix_PlayChannel(channel, sample_chunk, loops)
        if ret < 0:
            err = mixer.Mix_GetError().decode("utf8")
            raise RuntimeError("Error playing sample: {0}".format(err))

        return channel

    def halt_channel(self, channel:int) -> None:
        mixer.Mix_HaltChannel(channel)
