""" Characters and stuff characters do """

import abc
import logging
import enum
import uuid
import weakref
from typing import Optional, Any, Union, TYPE_CHECKING
from collections.abc import Mapping, MutableMapping, MutableSequence, Iterable

from stellarpunk import util, dialog
from .base import Entity, Sprite, EconAgent, Asset
from .sector_entity import SectorEntity

if TYPE_CHECKING:
    from .gamestate import Gamestate


class AgendumLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, character:"Character", *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.character = character

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        return f'{self.character.address_str()} {msg}', kwargs

class CharacterObserver(abc.ABC):
    def character_destroyed(self, character: "Character") -> None:
        pass

class Agendum:
    """ Represents an activity a Character is engaged in and how they can
    interact with the world. """

    def __init__(self, character:"Character", gamestate:"Gamestate") -> None:
        self.character = character
        self.gamestate = gamestate
        self.logger = AgendumLoggerAdapter(
                self.character,
                logging.getLogger(util.fullname(self)),
        )

        logging.getLogger(util.fullname(self))

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
        self._unpause()

    def pause(self) -> None:
        self._pause()
        self.gamestate.unschedule_agendum(self)

    def stop(self) -> None:
        self._stop()
        self.gamestate.unschedule_agendum(self)

    def is_complete(self) -> bool:
        return False

    def act(self) -> None:
        """ Lets the character interact. Called when scheduled. """
        pass

class Character(Entity):
    id_prefix = "CHR"
    def __init__(self, sprite:Sprite, *args:Any, home_sector_id:uuid.UUID, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.portrait:Sprite = sprite
        #TODO: other character background stuff

        #TODO: does location matter?
        self.location:Optional[SectorEntity] = None

        # how much money
        self.balance:float = 0.

        # owned assets (ships, stations)
        #TODO: are these actually SectorEntity instances? maybe a new co-class (Asset)
        self.assets:list[Asset] = []
        # activites this character is enaged in (how they interact)
        self.agenda:list[Agendum] = []

        self.observers:weakref.WeakSet[CharacterObserver] = weakref.WeakSet()

        self.home_sector_id = home_sector_id

    def destroy(self) -> None:
        super().destroy()
        for observer in self.observers.copy():
            observer.character_destroyed(self)
        self.observers.clear()
        for agendum in self.agenda:
            agendum.stop()
        self.location = None

    def observe(self, observer:CharacterObserver) -> None:
        self.observers.add(observer)

    def unobserve(self, observer:CharacterObserver) -> None:
        try:
            self.observers.remove(observer)
        except KeyError:
            pass

    def address_str(self) -> str:
        if self.location is None:
            return f'{self.short_id()}:None'
        else:
            return f'{self.short_id()}:{self.location.address_str()}'

    def take_ownership(self, asset:Asset) -> None:
        self.assets.append(asset)
        asset.owner = self

    def add_agendum(self, agendum:Agendum, start:bool=True) -> None:
        self.agenda.append(agendum)
        if start:
            agendum.start()

class AbstractEventManager:
    def e(self, event_id: enum.IntEnum) -> int:
        raise NotImplementedError()
    def ck(self, context_key: enum.IntEnum) -> int:
        raise NotImplementedError()
    def f(self, flag:str) -> int:
        raise NotImplementedError()
    def trigger_event(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: Mapping[int,int],
        event_args: dict[str, Union[int,float,str,bool]],
    ) -> None:
        raise NotImplementedError()

    def trigger_event_immediate(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: Mapping[int,int],
        event_args: dict[str, Union[int,float,str,bool]],
    ) -> None:
        raise NotImplementedError()

    def tick(self) -> None:
        raise NotImplementedError()


class Message(Entity):
    id_prefix = "MSG"

    def __init__(self, message_id:int, subject:str, message:str, timestamp:float, reply_to:uuid.UUID, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.message_id = message_id
        self.subject = subject
        self.message = message
        self.timestamp = timestamp

        self.reply_to = reply_to
        self.replied_at:Optional[float] = None


class Player(Entity):
    id_prefix = "PLR"

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # which character the player controls
        self._character:Optional[Character] = None # type: ignore[assignment]
        self.agent:EconAgent = None # type: ignore[assignment]

        self.messages:dict[uuid.UUID, Message] = {}

    @property
    def character(self) -> Optional[Character]:
        return self._character

    @character.setter
    def character(self, character:Optional[Character]) -> None:
        self._character = character

