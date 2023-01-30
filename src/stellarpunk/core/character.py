""" Characters and stuff characters do """

import abc
import logging
import uuid
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

    def start(self) -> None:
        self._start()

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
    def __init__(self, sprite:Sprite, location:SectorEntity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.portrait:Sprite = sprite
        #TODO: other character background stuff

        #TODO: does location matter?
        self.location:SectorEntity = location

        # how much money
        self.balance:float = 0.

        # owned assets (ships, stations)
        #TODO: are these actually SectorEntity instances? maybe a new co-class (Asset)
        self.assets:MutableSequence[Asset] = []
        # activites this character is enaged in (how they interact)
        self.agenda:MutableSequence[Agendum] = []

    def address_str(self) -> str:
        return f'{self.short_id()}:{self.location.address_str()}'

    def take_ownership(self, asset:Asset) -> None:
        self.assets.append(asset)
        asset.owner = self

    def add_agendum(self, agendum:Agendum, start:bool=True) -> None:
        self.agenda.append(agendum)
        if start:
            agendum.start()


class Message(Entity):
    id_prefix = "MSG"

    def __init__(self, message:str, timestamp:float, *args:Any, reply_to:Optional["Character"]=None, reply_dialog:Optional[dialog.DialogGraph]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.message = message
        self.timestamp = timestamp

        self.reply_to = reply_to
        self.reply_dialog = reply_dialog
        self.replied_at:Optional[float] = None


class PlayerObserver(abc.ABC):
    def notification_received(self, player:"Player", notification:str) -> None:
        pass
    def message_received(self, player:"Player", message:Message) -> None:
        pass


class Player(Entity):
    id_prefix = "PLR"

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.observers:Set[PlayerObserver] = set()

        # which character the player controls
        self.character:Character = None # type: ignore[assignment]
        self.agent:EconAgent = None # type: ignore[assignment]

        self.notifications:List[str] = []
        self.messages:Dict[uuid.UUID, Message] = {}

    def observe(self, observer:PlayerObserver) -> None:
        self.observers.add(observer)

    def send_notification(self, notification:str) -> None:
        self.notifications.append(notification)
        for observer in self.observers:
            observer.notification_received(self, notification)

    def send_message(self, message:Message) -> None:
        self.messages[message.entity_id] = message
        for observer in self.observers:
            observer.message_received(self, message)
