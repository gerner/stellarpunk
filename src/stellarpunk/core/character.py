""" Characters and stuff characters do """

import abc
import logging
import enum
import uuid
import weakref
from typing import Optional, Any, Union, Type
from collections.abc import Mapping, MutableMapping, MutableSequence, Iterable, Collection

import numpy as np

from stellarpunk import util, dialog
from . import base, sector

class Asset(base.Entity):
    """ An abc for classes that are assets ownable by characters. """
    def __init__(self, *args:Any, owner:Optional["Character"]=None, **kwargs:Any) -> None:
        # forward arguments onward, so implementing classes should inherit us
        # first
        super().__init__(*args, **kwargs)
        self.owner = owner

class IntelMatchCriteria:
    @abc.abstractmethod
    def matches(self, intel:"Intel") -> bool: ...

class IntelObserver(base.Observer):
    def intel_expired(self, intel:"Intel") -> None:
        pass

class Intel(base.Observable[IntelObserver], base.Entity):
    id_prefix = "INT"

    def __init__(self, *args:Any, expires_at:float=np.inf, fresh_until:Optional[float]=None, **kwargs:Any):
        # we set these fields before calling super init because they may be
        # referenced in overriden __str__ calls before we get to initialize
        # them.
        if not fresh_until:
            fresh_until = expires_at

        self.fresh_until = fresh_until
        self.expires_at = expires_at
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f'{self.short_id()} {type(self)} {self.is_valid()=} {self.is_fresh()=}'

    @abc.abstractmethod
    def match_criteria(self) -> IntelMatchCriteria: ...

    def sanity_check(self) -> None:
        pass

    @property
    def observable_id(self) -> uuid.UUID:
        return self.entity_id

    def _unobserve(self, observer:IntelObserver) -> None:
        # when we've got no one left observing us, we get nuked
        # this is equivalent to no one knowing (or caring) about this piece of
        # intel any more, so it cannot be transferred to another character
        if len(self.observers) == 0:
            self.entity_registry.destroy_entity(self)

    def matches(self, other:"Intel") -> bool:
        return False

    def is_valid(self) -> bool:
        return self.expires_at > self.entity_registry.now()

    def is_fresh(self) -> bool:
        return self.fresh_until > self.entity_registry.now()

    def expire(self) -> None:
        assert(not self.is_valid())
        for observer in list(self.observers):
            observer.intel_expired(self)
        self.clear_observers()
        # when the last observer is cleared we'll destroy ourselves
        # so destroying here is redundant
        # self.entity_registry.destroy_entity(self)

class IntelManagerObserver(base.Observer):
    def intel_added(self, intel_manager:"AbstractIntelManager", intel:"Intel") -> None:
        """ A piece of intel has been added. """
        pass
    def intel_removed(self, intel_manager:"AbstractIntelManager", intel:"Intel") -> None:
        """ A piece of intel has been removed. the intel may be "dead" """
        pass
    def intel_desired(self, intel_manager:"AbstractIntelManager", intel_criteria:"IntelMatchCriteria") -> None:
        """ Someone desires a particular kind of intel. """
        pass


class AbstractIntelManager(base.Observable[IntelManagerObserver]):
    @abc.abstractmethod
    def add_intel(self, intel:"Intel") -> None: ...
    @abc.abstractmethod
    def intel[T:Intel](self, match_criteria:IntelMatchCriteria, cls:Type[T]) -> Collection[T]: ...
    @abc.abstractmethod
    def get_intel[T:Intel](self, match_criteria:IntelMatchCriteria, cls:Type[T]) -> Optional[T]: ...
    @abc.abstractmethod
    def sanity_check(self) -> None: ...

    #TODO: some way to ask for asteroids
    #TODO: some way to ask for buyers (econ agents, or at least some proxy representing our knowledge of the econ agent at that time) and the corresponding locations we can find them at
    #TODO:

class AgendumLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, character:"Character", *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.character = character

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        return f'{self.character.address_str()} {msg}', kwargs

#TODO: should effects be entities? perhaps some kind of scheduled or action entity
class AbstractAgendum(abc.ABC):
    """ Represents an activity a Character is engaged in and how they can
    interact with the world. """

    @classmethod
    def create_agendum[T:AbstractAgendum](cls:Type[T], character:"Character", *args:Any, **kwargs:Any) -> T:
        kwargs.update({"_check_flag":True})
        agendum = cls(*args, **kwargs)
        agendum.initialize_agendum(character)
        return agendum

    def __init__(self, agenda_id:Optional[uuid.UUID]=None, _check_flag:bool=False) -> None:
        assert(_check_flag is True)
        if agenda_id:
            assert(isinstance(agenda_id, uuid.UUID))
            self.agenda_id = agenda_id
        else:
            self.agenda_id = uuid.uuid4()
        self.character:Character = None # type: ignore
        self.logger:AgendumLoggerAdapter = None # type: ignore
        self.paused = False
        self._is_primary = False

    def initialize_agendum(self, character:"Character") -> None:
        self.character = character
        self.logger = AgendumLoggerAdapter(
                self.character,
                logging.getLogger(util.fullname(self)),
        )

    def is_primary(self) -> bool:
        return self._is_primary

    def make_primary(self) -> None:
        self._is_primary = True

    def preempt_primary(self) -> None:
        assert(self._is_primary)
        self._is_primary = False

    def find_primary(self) -> Optional["AbstractAgendum"]:
        for agendum in self.character.agenda:
            if agendum.is_primary():
                return agendum
        return None

    def sanity_check(self) -> None:
        assert(self in self.character.agenda)

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def _unpause(self) -> None:
        pass

    def _pause(self) -> None:
        pass

    def start(self) -> None:
        self._start()

    def unpause(self) -> None:
        self.paused = False
        self._unpause()

    def pause(self) -> None:
        self.paused = True
        self._pause()

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def register(self) -> None: ...

    @abc.abstractmethod
    def unregister(self) -> None: ...

    def is_complete(self) -> bool:
        return False

    def act(self) -> None:
        """ Lets the character interact. Called when scheduled. """
        pass

class CharacterObserver(base.Observer, abc.ABC):
    def character_destroyed(self, character: "Character") -> None:
        pass

class Character(base.Observable[CharacterObserver], base.Entity):
    id_prefix = "CHR"

    @classmethod
    def create_character[T:"Character"](cls:Type[T], sprite:base.Sprite, intel_manager:AbstractIntelManager, *args:Any, **kwargs:Any) -> T:
        character = cls(sprite, *args, _check_flag=True, **kwargs)
        character.intel_manager = intel_manager
        return character

    def __init__(self, sprite:base.Sprite, *args:Any, home_sector_id:uuid.UUID, _check_flag:bool=False, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.portrait:base.Sprite = sprite
        #TODO: other character background stuff

        #TODO: does location matter?
        self.location:Optional[sector.SectorEntity] = None

        # how much money
        self.balance:float = 0.

        # owned assets (ships, stations)
        #TODO: are these actually SectorEntity instances? maybe a new co-class (Asset)
        self.assets:list[Asset] = []
        # activites this character is enaged in (how they interact)
        self.agenda:list[AbstractAgendum] = []

        self.home_sector_id = home_sector_id

        self.intel_manager:AbstractIntelManager = None # type: ignore

    def sanity_check(self) -> None:
        super().sanity_check()
        self.intel_manager.sanity_check()

    # base.Observable
    @property
    def observable_id(self) -> uuid.UUID:
        return self.entity_id


    def destroy(self) -> None:
        super().destroy()
        for observer in self._observers.copy():
            observer.character_destroyed(self)
        self.clear_observers()
        for agendum in self.agenda:
            agendum.stop()
        self.location = None

    def address_str(self) -> str:
        if self.location is None:
            return f'{self.short_id()}:None'
        else:
            return f'{self.short_id()}:{self.location.address_str()}'

    def take_ownership(self, asset:Asset) -> None:
        self.assets.append(asset)
        asset.owner = self

    def add_agendum(self, agendum:AbstractAgendum, start:bool=True) -> None:
        agendum.register()
        self.agenda.append(agendum)
        if start:
            agendum.start()

    def remove_agendum(self, agendum:AbstractAgendum) -> None:
        self.agenda.remove(agendum)
        agendum.unregister()

class AbstractEventManager:
    def e(self, event_id: enum.IntEnum) -> int:
        raise NotImplementedError()
    def ck(self, context_key: enum.IntEnum) -> int:
        raise NotImplementedError()
    def f(self, flag:str) -> int:
        raise NotImplementedError()

    def e_rev(self, event_id:int) -> str:
        raise NotImplementedError()
    def ck_rev(self, context_key:int) -> str:
        raise NotImplementedError()

    def trigger_event(
        self,
        characters: Collection[Character],
        event_type: int,
        context: dict[int,int],
        event_args: dict[str, Union[int,float,str,bool]],
        merge_key: Optional[uuid.UUID]=None,
    ) -> None:
        raise NotImplementedError()

    def trigger_event_immediate(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: dict[int,int],
        event_args: dict[str, Union[int,float,str,bool]],
    ) -> None:
        raise NotImplementedError()

    def tick(self) -> None:
        raise NotImplementedError()


class Message(base.Entity):
    id_prefix = "MSG"

    def __init__(self, message_id:int, subject:str, message:str, timestamp:float, reply_to:uuid.UUID, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.message_id = message_id
        self.subject = subject
        self.message = message
        self.timestamp = timestamp

        self.reply_to = reply_to
        self.replied_at:Optional[float] = None


class Player(base.Entity):
    id_prefix = "PLR"

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # which character the player controls
        self._character:Optional[Character] = None # type: ignore[assignment]
        self.agent:base.EconAgent = None # type: ignore[assignment]

        self.messages:dict[uuid.UUID, Message] = {}

    @property
    def character(self) -> Optional[Character]:
        return self._character

    @character.setter
    def character(self, character:Optional[Character]) -> None:
        self._character = character

class CrewedSectorEntity(sector.SectorEntity):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.captain: Optional["Character"] = None

def captain(craft:sector.SectorEntity) -> Optional[Character]:
    if isinstance(craft, CrewedSectorEntity) and craft.captain:
        return craft.captain
    return None
