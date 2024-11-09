""" Characters and stuff characters do """

import abc
import logging
import uuid
import weakref
from typing import Optional, Any, MutableSequence, Set, List, Dict, TYPE_CHECKING

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
    def message_received(self, character: "Character", message: "Message") -> None:
        pass

    def character_destroyed(self, character: "Character") -> None:
        pass

class Agendum(CharacterObserver):
    """ Represents an activity a Character is engaged in and how they can
    interact with the world. """

    def __init__(self, character:"Character", gamestate:"Gamestate") -> None:
        self.character = character
        self.character.observe(self)
        self.gamestate = gamestate
        self.logger = AgendumLoggerAdapter(
                self.character,
                logging.getLogger(util.fullname(self)),
        )

        logging.getLogger(util.fullname(self))

    def character_destroyed(self, character:"Character") -> None:
        self.stop()

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
        self.character.unobserve(self)
        self.gamestate.unschedule_agendum(self)

    def is_complete(self) -> bool:
        return False

    def act(self) -> None:
        """ Lets the character interact. Called when scheduled. """
        pass

class Character(Entity):
    id_prefix = "CHR"
    def __init__(self, sprite:Sprite, location:SectorEntity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.portrait:Sprite = sprite
        #TODO: other character background stuff

        #TODO: does location matter?
        self.location:Optional[SectorEntity] = location

        # how much money
        self.balance:float = 0.

        # owned assets (ships, stations)
        #TODO: are these actually SectorEntity instances? maybe a new co-class (Asset)
        self.assets:MutableSequence[Asset] = []
        # activites this character is enaged in (how they interact)
        self.agenda:MutableSequence[Agendum] = []

        self.observers:Set[CharacterObserver] = set()

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

    def send_message(self, message: "Message") -> None:
        for observer in self.observers:
            observer.message_received(self, message)


class Message(Entity):
    id_prefix = "MSG"

    def __init__(self, message_id:int, subject:str, message:str, timestamp:float, reply_to:"Character", *args:Any, reply_dialog:Optional[dialog.DialogGraph]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.message_id = message_id
        self.subject = subject
        self.message = message
        self.timestamp = timestamp

        self.reply_to = reply_to
        self.reply_dialog = reply_dialog
        self.replied_at:Optional[float] = None


class Player(Entity, CharacterObserver):
    id_prefix = "PLR"

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # which character the player controls
        self._character:Character = None # type: ignore[assignment]
        self.agent:EconAgent = None # type: ignore[assignment]

        self.messages:Dict[uuid.UUID, Message] = {}

    def get_character(self) -> Character:
        return self._character

    def set_character(self, character:Character) -> None:
        if self._character:
            self._character.unobserve(self)

        self._character = character
        self._character.observe(self)

    character = property(get_character, set_character, doc="which character this player plays as")

    def message_received(self, character:Character, message:Message) -> None:
        self.messages[message.entity_id] = message
