import time
import io
import os
import glob
import abc
import enum
import pydoc
import uuid
import datetime
import logging
import contextlib
import tempfile
import itertools
import weakref
import collections
from typing import Any, Optional, TypeVar, Union, Type, Callable

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, sim, core, narrative, generate, econ, events
from stellarpunk.agenda import intel as aintel
from stellarpunk.serialization import util as s_util

SAVE_FORMAT_VERSION = "0.1.0"

class LoadContext:
    """ State for one load cycle. """

    def __init__(self, sg:"GameSaver") -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.save_game = sg
        self.debug = False
        self.generator:generate.UniverseGenerator = sg.generator
        self.gamestate:core.Gamestate = None # type: ignore
        self.references:set[Any] = set()
        self.ephemeral_fetch_store:dict[uuid.UUID, Any] = dict()
        self._post_loads:list[tuple[Any, Any]] = []
        self._sanity_checks:list[tuple[Any, Any]] = []
        self._observer_sanity_checks:list[tuple[core.Observable, list[tuple[str, uuid.UUID]]]] = []
        self._custom_post_loads:list[tuple[Callable[[Any, LoadContext, Any], None], Any, Any]] = []
        self._custom_sanity_checks:list[tuple[Callable[[Any, LoadContext, Any], None], Any, Any]] = []

    def debug_string_r(self, s:str, f:io.IOBase) -> str:
        if self.debug:
            return s_util._debug_string_r(s, f)
        return s

    def reference(self, obj:Any) -> None:
        """ hang on to a reference to an object.

        useful when the only collection in first load pass is holding weakrefs.
        this way these objects will stay alive until this LoadContext goes away
        """
        self.references.add(obj)

    def store(self, object_id:uuid.UUID, obj:Any) -> None:
        """ hang on to an object for later fetch during this load cycle. """
        self.ephemeral_fetch_store[object_id] = obj

    def fetch[T](self, object_id:uuid.UUID, klass:Type[T]) -> T:
        obj = self.ephemeral_fetch_store[object_id]
        assert(isinstance(obj, klass))
        return obj

    def register_post_load(self, obj:Any, context:Any) -> None:
        self._post_loads.append((obj, context))

    def register_custom_post_load(self, fn:Callable[[Any, "LoadContext", Any], None], obj:Any, context:Any) -> None:
        self._custom_post_loads.append((fn, obj, context))

    def register_sanity_check(self, obj:Any, context:Any) -> None:
        self._sanity_checks.append((obj, context))

    def register_custom_sanity_check(self, fn:Callable[[Any, "LoadContext", Any], None], obj:Any, context:Any) -> None:
        self._custom_sanity_checks.append((fn, obj, context))

    def register_sanity_check_observers(self, obj:core.Observable, observer_ids:list[tuple[str, uuid.UUID]]) -> None:
        self._observer_sanity_checks.append((obj, observer_ids))

    def load_complete(self) -> None:
        self.logger.info("post loading...")
        for obj, context in self._post_loads:
            self.save_game.post_load_object(obj, self, context)

        for fn, obj, context in self._custom_post_loads:
            fn(obj, self, context)

        self.logger.info("sanity checking...")
        for obj, context in self._sanity_checks:
            self.save_game.sanity_check_object(obj, self, context)

        for fn, obj, context in self._custom_sanity_checks:
            fn(obj, self, context)

        self.logger.info("sanity checking observers...")
        for obj, context in self._observer_sanity_checks:
            self.save_game.sanity_check_observers(obj, self, context)

class LoadErrorCase(enum.Enum):
    ABORT = enum.auto()

class LoadError(Exception):
    def __init__(self, case:LoadErrorCase, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.case = case

class SaverObserver:
    def load_tick(self, saver:"Saver") -> None:
        pass
    def save_tick(self, saver:"Saver") -> None:
        pass

class Saver[T](abc.ABC):
    def __init__(self, save_game:"GameSaver"):
        self.save_game = save_game
        self._observers:weakref.WeakSet[SaverObserver] = weakref.WeakSet()

    @abc.abstractmethod
    def save(self, obj:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def load(self, f:io.IOBase, load_context:LoadContext) -> T: ...
    def fetch(self, klass:Type[T], object_id:uuid.UUID, load_context:LoadContext) -> T:
        raise NotImplementedError()

    def save_object(self, obj:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written = self.save(obj, f)
        if isinstance(obj, core.Observer):
            bytes_written += self.save_observing(obj, f)
        if isinstance(obj, core.Observable):
            bytes_written += self.save_observers(obj, f)
        return bytes_written

    def load_object(self, f:io.IOBase, load_context:LoadContext) -> T:
        obj = self.load(f, load_context)
        if isinstance(obj, core.Observer):
            self.load_observing(obj, f, load_context)
        if isinstance(obj, core.Observable):
            self.load_observers(obj, f, load_context)
        return obj

    def save_observers(self, obj:core.Observable, f:io.IOBase) -> int:
        bytes_written = 0
        if self.save_game.debug:
            for observer in obj.observers:
                assert(obj in observer.observings)

            bytes_written += self.save_game.debug_string_w("observers", f)
            # skip ephemeral observers with sentinel observer id
            bytes_written += s_util.str_uuids_to_f(list((util.fullname(x), x.observer_id) for x in obj.observers if x.observer_id != core.OBSERVER_ID_NULL), f)
        return bytes_written

    def load_observers(self, obj:core.Observable, f:io.IOBase, load_context:LoadContext) -> list[tuple[str, uuid.UUID]]:
        observer_ids:list[tuple[str, uuid.UUID]] = []
        if load_context.debug:
            load_context.debug_string_r("observers", f)
            observer_ids = s_util.str_uuids_from_f(f)
            load_context.register_sanity_check_observers(obj, observer_ids)
        return observer_ids

    def save_observing(self, obj:core.Observer, f:io.IOBase) -> int:
        """ Saves Observable instances this obj is observing. """

        if self.save_game.debug:
            for observable in obj.observings:
                assert(obj in observable.observers)

        bytes_written = 0
        bytes_written += self.save_game.debug_string_w("observing", f)
        bytes_written += s_util.str_uuids_to_f(list((util.fullname(x), x.observable_id) for x in obj.observings), f)
        return bytes_written

    def load_observing(self, obj:core.Observer, f:io.IOBase, load_context:LoadContext) -> list[tuple[type, uuid.UUID]]:
        """ Restores info about Observable instances this obj is observing. """
        load_context.debug_string_r("observing", f)
        raw_observing_info = s_util.str_uuids_from_f(f)
        observing_info:list[tuple[type, uuid.UUID]] = []
        for klassname, object_id in raw_observing_info:
            klass = pydoc.locate(klassname)
            assert(isinstance(klass, type))
            assert(issubclass(klass, core.Observable))
            observing_info.append((klass, object_id))
        load_context.register_custom_post_load(self.post_load_observing, obj, observing_info)
        return observing_info

    def post_load_observing(self, observer:core.Observer, load_context:LoadContext, observing:list[tuple[type, uuid.UUID]]) -> None:
        """ Actually sets up obj observing Observables it observed at save. """
        for klass, object_id in observing:
            observable:core.Observable = self.save_game.fetch_object(klass, object_id, load_context)
            assert(isinstance(observable, core.Observable))
            observable.observe(observer)

    def post_load(self, obj:T, load_context:LoadContext, context:Any) -> None:
        pass

    def sanity_check_observers(self, obj:core.Observable, load_context:LoadContext, observer_ids:list[tuple[str, uuid.UUID]]) -> None:
        # make sure all the observers we had when saving are back
        saved_observer_counts:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for observer_id in observer_ids:
            saved_observer_counts[observer_id] += 1
        loaded_observer_counts:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for observer in obj.observers:
            loaded_observer_counts[(util.fullname(observer), observer.observer_id)] += 1
        saved_observer_counts.subtract(loaded_observer_counts)
        non_zero_observers = {observer_id: count for observer_id, count in saved_observer_counts.items() if count != 0}
        assert(non_zero_observers == {})

    def sanity_check(self, obj:T, load_context:LoadContext, context:Any) -> None:
        pass

    def estimate_ticks(self, obj:T) -> int:
        return 0

    def observe(self, observer:SaverObserver) -> None:
        self._observers.add(observer)

    def unobserve(self, observer:SaverObserver) -> None:
        self._observers.remove(observer)

    def load_tick(self) -> None:
        for observer in self._observers:
            observer.load_tick(self)

    def save_tick(self) -> None:
        for observer in self._observers:
            observer.save_tick(self)

class NoneSaver(Saver[None]):
    def save(self, obj:None, f:io.IOBase) -> int:
        return 0
    def load(self, f:io.IOBase, load_context:LoadContext) -> None:
        return None

class SaveGame:
    def __init__(self, save_format_version:str, game_version:str, game_start_version:str, debug_flag:bool, save_date:datetime.datetime, estimated_ticks:int, game_fingerprint:bytes, game_timestamp:float, game_base_date:datetime.datetime, game_secs_per_sec:float, game_save_count:int, pc_name:str, pc_sector_name:str, filename:str=""):
        self.filename = filename
        self.save_format_version = SAVE_FORMAT_VERSION
        self.game_version = game_version
        self.game_start_version = game_start_version
        self.debug_flag = debug_flag
        self.save_date = save_date
        self.estimated_ticks = estimated_ticks

        self.game_fingerprint = game_fingerprint
        self.game_timestamp = game_timestamp
        self.game_base_date = game_base_date
        self.game_secs_per_sec = game_secs_per_sec
        self.game_save_count = game_save_count

        self.pc_name = pc_name
        self.pc_sector_name = pc_sector_name

    @property
    def game_date(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.game_base_date.timestamp() + self.game_timestamp*self.game_secs_per_sec)


class GameSaverObserver:
    def load_start(self, estimated_ticks:int, game_saver:"GameSaver") -> None:
        pass
    def load_tick(self, game_saver:"GameSaver") -> None:
        pass
    def load_complete(self, load_context:LoadContext, game_saver:"GameSaver") -> None:
        pass


    def save_start(self, estimated_ticks:int, game_saver:"GameSaver") -> None:
        pass
    def save_tick(self, game_saver:"GameSaver") -> None:
        pass
    def save_complete(self, game_saver:"GameSaver") -> None:
        pass

class GameSaver(SaverObserver):
    """ Central point for saving a game.

    Organizes configuration and dispatches to various sorts of save logic. """

    def __init__(self, generator:generate.UniverseGenerator, event_manager:events.EventManager, intel_director:aintel.IntelCollectionDirector, *args:Any, debug:bool=True, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(util.fullname(self))

        #TODO: how do we want to handle the save file format? increment when we change the format? always tag with code version?
        self.save_format_version = core.stellarpunk_version()

        self.debug = debug
        self._save_path:str = "/tmp/stellarpunk_saves"
        self._save_file_glob = "save_*.stpnk"
        #TODO: multiple autosaves?
        self._autosave_glob = "autosave.stpnk"
        self._save_register:dict[type, Saver] = {}
        self._class_key_lookup:dict[type, int] = {}
        self._key_class_lookup:dict[int, type] = {}

        # we'll need these as part of saving and loading
        self.generator = generator
        self.event_manager = event_manager
        self.intel_director = intel_director

        self._observers:weakref.WeakSet[GameSaverObserver] = weakref.WeakSet()

    def debug_string_w(self, s:str, f:io.IOBase) -> int:
        if self.debug:
            return s_util._debug_string_w(s, f)
        return 0

    def observe(self, observer:GameSaverObserver) -> None:
        self._observers.add(observer)

    def unobserve(self, observer:GameSaverObserver) -> None:
        self._observers.remove(observer)

    # SaverObserver
    def load_tick(self, saver:Saver) -> None:
        for observer in self._observers:
            observer.load_tick(self)

    def save_tick(self, saver:Saver) -> None:
        for observer in self._observers:
            observer.save_tick(self)

    def _gen_save_filename(self) -> str:
        return f'save_{time.time()}.stpnk'

    def _save_registry(self, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.size_to_f(len(self._class_key_lookup), f)
        for klass, key in self._class_key_lookup.items():
            bytes_written += s_util.to_len_pre_f(util.fullname(klass), f)
            bytes_written += s_util.int_to_f(key, f)
        return bytes_written

    def _load_registry(self, f:io.IOBase) -> None:
        # all the savers must be registered already, we're just loading the
        # class -> key correspondence the file originally used when saving
        # we want to verify that we have all the right savers registered and
        # that they have the same id keys
        # note that in general it's ok for us to have some extra savers
        count = s_util.size_from_f(f)
        for _ in range(count):
            fullname = s_util.from_len_pre_f(f)
            key = s_util.int_from_f(f)
            klass = pydoc.locate(fullname)
            assert(isinstance(klass, type))

            #TODO: should we just stomp on the registry?
            # if this doesn't match we might have a problem as it implies logic
            # has changed since we saved
            assert(self._class_key_lookup[klass] == key)

        #TODO: we probably don't want to assert this. we might add new types in
        # new versions of the code, but that doesn't necessarily invalidate
        # prior savegames that didn't know about these new types
        assert(count == len(self._class_key_lookup))

    def _save_metadata(self, gamestate:core.Gamestate, save_file:io.IOBase) -> int:
        bytes_written = 0

        bytes_written += s_util.to_len_pre_f(self.save_format_version, save_file)
        bytes_written += s_util.to_len_pre_f(gamestate.game_version, save_file)
        bytes_written += s_util.to_len_pre_f(gamestate.game_start_version, save_file)

        bytes_written += s_util.bool_to_f(self.debug, save_file)

        bytes_written += s_util.to_len_pre_f(datetime.datetime.now().isoformat(), save_file)
        estimated_ticks = self.estimate_ticks(gamestate)
        bytes_written += s_util.int_to_f(estimated_ticks, save_file)

        bytes_written += s_util.bytes_to_f(gamestate.fingerprint, save_file)
        bytes_written += s_util.float_to_f(gamestate.timestamp, save_file)
        bytes_written += s_util.to_len_pre_f(gamestate.base_date.isoformat(), save_file)
        bytes_written += s_util.float_to_f(gamestate.game_secs_per_sec, save_file)
        bytes_written += s_util.int_to_f(gamestate.save_count, save_file)

        # TODO: will the player always have selected a character and be in a sector?
        if gamestate.player.character:
            bytes_written += s_util.to_len_pre_f(gamestate.player.character.name, save_file)
            if gamestate.player.character.location:
                bytes_written += s_util.to_len_pre_f(gamestate.player.character.location.name, save_file)
            else:
                bytes_written += s_util.to_len_pre_f("", save_file)
        else:
            bytes_written += s_util.to_len_pre_f("", save_file)
            bytes_written += s_util.to_len_pre_f("", save_file)

        return bytes_written

    def _load_metadata(self, save_file:io.IOBase) -> SaveGame:
        save_format_version = s_util.from_len_pre_f(save_file)
        game_version = s_util.from_len_pre_f(save_file)
        game_start_version = s_util.from_len_pre_f(save_file)

        debug_flag = s_util.bool_from_f(save_file)
        save_date = datetime.datetime.fromisoformat(s_util.from_len_pre_f(save_file))
        estimated_ticks = s_util.int_from_f(save_file)
        for observer in self._observers:
            observer.load_start(estimated_ticks, self)

        fingerprint = s_util.bytes_from_f(save_file)
        timestamp = s_util.float_from_f(save_file)
        base_date = datetime.datetime.fromisoformat(s_util.from_len_pre_f(save_file))
        game_secs_per_sec = s_util.float_from_f(save_file)
        save_count = s_util.int_from_f(save_file)

        pc_name = s_util.from_len_pre_f(save_file)
        pc_sector_name = s_util.from_len_pre_f(save_file)

        return SaveGame(save_format_version, game_version, game_start_version, debug_flag, save_date, estimated_ticks, fingerprint, timestamp, base_date, game_secs_per_sec, save_count, pc_name, pc_sector_name)

    def key_from_class(self, klass:type) -> int:
        return self._class_key_lookup[klass]

    def class_from_key(self, key:int) -> type:
        return self._key_class_lookup[key]

    def register_saver(self, klass:type, saver:Saver) -> int:
        """ Registers type specific save/load logic

        returns the key used to identify that type in save files """
        self._save_register[klass] = saver
        key = len(self._class_key_lookup)
        self._class_key_lookup[klass] = key
        self._key_class_lookup[key] = klass
        saver.observe(self)
        return key

    def ignore_saver(self, klass:type) -> int:
        return self.register_saver(klass, NoneSaver(self))

    def estimate_ticks(self, obj:Any, klass:Optional[type]=None) -> int:
        if klass is None:
            klass = type(obj)
        else:
            assert(isinstance(obj, klass))

        return self._save_register[klass].estimate_ticks(obj)

    def save_object(self, obj:Any, f:io.IOBase, klass:Optional[type]=None) -> int:
        if klass is None:
            klass = type(obj)
        else:
            assert(isinstance(obj, klass))

        bytes_written = 0
        if self.debug:
            bytes_written = s_util.to_len_pre_f(f'__so:{util.fullname(klass)}', f)
            bytes_written = s_util.to_len_pre_f(f'__so:{util.fullname(obj)}', f)
        return bytes_written+self._save_register[klass].save_object(obj, f)

    def load_object(self, klass:type, f:io.IOBase, load_context:LoadContext) -> Any:
        if load_context.debug:
            klassname = s_util.from_len_pre_f(f)
            fullname = s_util.from_len_pre_f(f)
            if klassname != f'__so:{util.fullname(klass)}':
                raise ValueError(f'{klassname} not __so:{util.fullname(klass)} at {f.tell()}')
        return self._save_register[klass].load_object(f, load_context)

    def fetch_object[T](self, klass:Type[T], object_id:uuid.UUID, load_context:LoadContext) -> T:
        return self._save_register[klass].fetch(klass, object_id, load_context)

    def post_load_object(self, obj:Any, load_context:LoadContext, context:Any) -> None:
        self._save_register[type(obj)].post_load(obj, load_context, context)

    def sanity_check_object(self, obj:Any, load_context:LoadContext, context:Any) -> None:
        self._save_register[type(obj)].sanity_check(obj, load_context, context)

    def sanity_check_observers(self, obj:Any, load_context:LoadContext, context:Any) -> None:
        self._save_register[type(obj)].sanity_check_observers(obj, load_context, context)

    def autosave(self, gamestate:core.Gamestate) -> str:
        #TODO: should we keep old autosaves?
        save_filename = "autosave.stpnk"
        save_filename = os.path.join(self._save_path, save_filename)
        return self.save(gamestate, save_filename)

    def save(self, gamestate:core.Gamestate, save_filename:Optional[str]=None, force_pause:bool=True) -> str:
        if force_pause:
            gamestate.force_pause(self)
        else:
            assert gamestate.paused

        self.logger.info(f'saving gamestate at {gamestate.timestamp} with {gamestate.ticks} ticks...')
        start_time = time.perf_counter()
        # plus one estimated tick for the sanity check
        estimated_ticks = self.estimate_ticks(gamestate)+1
        for observer in self._observers:
            observer.save_start(estimated_ticks, self)

        self.logger.debug("sanity checking gamestate")
        gamestate.sanity_check()
        for observer in self._observers:
            observer.save_tick(self)
        self.logger.debug("sanity check complete")

        # keep track of how many times we've saved this game
        # this will get restored on load
        gamestate.save_count += 1

        if save_filename is None:
            save_filename = self._gen_save_filename()
            save_filename = os.path.join(self._save_path, save_filename)
        bytes_written = 0
        with contextlib.ExitStack() as context_stack:
            temp_save_file = context_stack.enter_context(tempfile.NamedTemporaryFile("wb", delete=not self.debug))
            temp_name = temp_save_file.name
            self.logger.debug(f'saving to temp file {temp_save_file.name}')
            save_file:io.IOBase = temp_save_file # type: ignore
            # put metadata about the save game at the top for quick retrieval
            bytes_written += self._save_metadata(gamestate, save_file)

            # save class -> key registration
            bytes_written += self.debug_string_w("class registry", save_file)
            bytes_written += self._save_registry(save_file)

            #TODO: put some global state error checking stuff. these are things
            # we don't actually save, but game state references like config
            # items. this helps avoid inconsistencies down the road as
            # code/config changes between save/load
            # e.g. sprites, cultures, event context keys

            bytes_written += self.debug_string_w("event state", save_file)
            #TODO: shouldn't we be saving it off of gamestate and not us?
            # this is hard because Gamestate has an AbstractEventManager which
            # doesn't (can't) expose EventState
            bytes_written += self.save_object(self.event_manager.event_state, save_file)

            # save the simulator which will recursively save everything
            bytes_written += self.debug_string_w("gamestate", save_file)
            bytes_written += self.save_object(gamestate, save_file)

            # move the temp file into final home, so we only end up with good files
            os.rename(temp_save_file.name, save_filename)

        for observer in self._observers:
            observer.save_complete(self)

        self.logger.info(f'saved {bytes_written} bytes to {save_filename} in {time.perf_counter()-start_time}s')

        if force_pause:
            gamestate.force_unpause(self)

        return save_filename

    def list_save_games(self) -> list[SaveGame]:
        save_games = []
        for x in set(itertools.chain(
            glob.glob(os.path.join(self._save_path, self._save_file_glob)),
            glob.glob(os.path.join(self._save_path, self._autosave_glob))
        )):
            try:
                with open(x, "rb") as f:
                    save_game = self._load_metadata(f)
                    save_game.filename = x
                    save_games.append(save_game)
            except Exception as e:
                self.logger.error('ignoring bad save file: {x}')
        save_games.sort(key=lambda x: x.save_date, reverse=True)
        return save_games

    def load(self, save_filename:str, save_file:Optional[io.IOBase]=None) -> core.Gamestate:
        self.logger.info(f'loading {save_filename}')
        load_context = LoadContext(self)
        with contextlib.ExitStack() as context_stack:
            if save_file is None:
                save_file = context_stack.enter_context(open(save_filename, "rb"))
            self.logger.debug("loading metadata")
            save_game = self._load_metadata(save_file)
            load_context.debug = save_game.debug_flag

            # load the class -> key registration
            self.logger.debug("loading class registry")
            load_context.debug_string_r("class registry", save_file)
            self._load_registry(save_file)

            self.logger.debug("loading event state")
            load_context.debug_string_r("event state", save_file)
            event_state = self.load_object(events.EventState, save_file, load_context)
            self.logger.debug("loading gamestate")
            load_context.debug_string_r("gamestate", save_file)
            gamestate = self.load_object(core.Gamestate, save_file, load_context)
            gamestate.fingerprint = save_game.game_fingerprint
            # do not reset game_version, the game is now running under that
            # version and it got set in Gamestate constructor
            gamestate.game_start_version = save_game.game_start_version
            gamestate.save_count = save_game.game_save_count
            gamestate.event_manager = self.event_manager

            # final set up
            self.logger.debug("first pass load complete")
            load_context.load_complete()

            # we created the gamestate so it's our responsibility to set its
            # event manager and post_initialize it
            self.logger.debug("initializing event manager, etc.")
            gamestate.event_manager.initialize_gamestate(event_state, gamestate)
            self.intel_director.initialize_gamestate(gamestate)

            self.logger.debug("sanity checking loaded gamestate")
            gamestate.sanity_check()

            self.logger.debug("loading gamestate into generator")
            self.generator.load_universe(gamestate)

            for observer in self._observers:
                observer.load_complete(load_context, self)

            self.logger.info("load complete")
            return gamestate

class DispatchSaver[T](Saver[T]):
    """ Saves and loads Entities by dispatching to class specific logic.

    Gamestate doesn't know the type of entity when loading, so this saver reads
    a type code it writes during saving. all other saving/loading logic is
    dispatched to class specific logic. """

    def save(self, entity:T, f:io.IOBase) -> int:
        bytes_written = 0

        # save key so we know what type of entity to load!
        class_id = self.save_game.key_from_class(type(entity))
        bytes_written += s_util.int_to_f(class_id, f)

        # dispatch to specific entity type to save the rest
        bytes_written += self.save_game.save_object(entity, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:LoadContext) -> T:
        # read entity type
        class_id = s_util.int_from_f(f)
        klass = self.save_game.class_from_key(class_id)

        # dispatch to specific type
        entity = self.save_game.load_object(klass, f, load_context)

        return entity
