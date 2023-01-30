""" Stellarpunk core data model basic objects

No dependencies on other parts of the datamodel
"""

import enum
import abc
import uuid
import itertools
from typing import Optional, Tuple, List, Sequence, Dict, Any, Collection, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from stellarpunk.narrative import director

if TYPE_CHECKING:
    from .character import Character

class EntityRegistry(abc.ABC):
    @abc.abstractmethod
    def register_entity(self, entity: "Entity") -> None: ...

    @abc.abstractmethod
    def unregister_entity(self, entity: "Entity") -> None: ...

class Entity(abc.ABC):
    id_prefix = "ENT"

    def __init__(self, entity_registry: EntityRegistry, name:Optional[str]=None, entity_id:Optional[uuid.UUID]=None, description:Optional[str]=None)->None:
        self.entity_id = entity_id or uuid.uuid4()
        self._entity_id_short_int = int.from_bytes(self.entity_id.bytes[0:8], byteorder='big')

        if name is None:
            name = f'{self.__class__} {str(self.entity_id)}'
        self.name = name

        self.description = description or name

        self.context = director.EventContext()

        self.entity_registry = entity_registry
        self.entity_registry.register_entity(self)

    def __del__(self) -> None:
        self.entity_registry.unregister_entity(self)

    def short_id(self) -> str:
        """ first 32 bits as hex """
        return f'{self.id_prefix}-{self.entity_id.hex[:8]}'

    def short_id_int(self) -> int:
        return self._entity_id_short_int

    def __str__(self) -> str:
        return f'{self.short_id()}'


class Asset(Entity):
    """ An abc for classes that are assets ownable by characters. """
    def __init__(self, *args:Any, owner:Optional["Character"]=None, **kwargs:Any) -> None:
        # forward arguments onward, so implementing classes should inherit us
        # first
        super().__init__(*args, **kwargs)
        self.owner = owner


class Sprite:
    """ A "sprite" from a text file that can be drawn in text """

    @staticmethod
    def load_sprites(data:str, sprite_size:Tuple[int, int]) -> List["Sprite"]:
        sprites = []
        sheet = data.split("\n")
        offset_limit = (len(sheet[0])//sprite_size[0], len(sheet)//sprite_size[1])
        for offset_x, offset_y in itertools.product(range(offset_limit[0]), range(offset_limit[1])):
            sprites.append(Sprite(
                [
                    x[offset_x*sprite_size[0]:offset_x*sprite_size[0]+sprite_size[0]] for x in sheet[offset_y*sprite_size[1]:offset_y*sprite_size[1]+sprite_size[1]]
                ]
            ))

        return sprites

    def __init__(self, text:Sequence[str], attr:Optional[Dict[Tuple[int,int], Tuple[int, int]]]=None) -> None:
        self.text = text
        self.attr = attr or {}
        self.height = len(text)
        if len(text) > 0:
            self.width = len(text[0])
        else:
            self.width = 0


class EconAgent(abc.ABC):
    _next_id = 0

    @classmethod
    def num_agents(cls) -> int:
        return EconAgent._next_id

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_id = EconAgent._next_id
        EconAgent._next_id += 1

    #@abc.abstractmethod
    #def get_owner(self) -> Character: ...

    #def get_character(self) -> Character:
    #    return self.get_owner()

    @abc.abstractmethod
    def buy_resources(self) -> Collection[int]: ...

    @abc.abstractmethod
    def sell_resources(self) -> Collection[int]: ...

    @abc.abstractmethod
    def buy_price(self, resource:int) -> float: ...

    @abc.abstractmethod
    def sell_price(self, resource:int) -> float: ...

    @abc.abstractmethod
    def balance(self) -> float: ...

    @abc.abstractmethod
    def budget(self, resource:int) -> float: ...

    @abc.abstractmethod
    def inventory(self, resource:int) -> float: ...

    @abc.abstractmethod
    def buy(self, resource:int, price:float, amount:float) -> None: ...

    @abc.abstractmethod
    def sell(self, resource:int, price:float, amount:float) -> None: ...

class AbstractEconDataLogger:
    def transact(self, diff:float, product_id:int, buyer:int, seller:int, price:float, sale_amount:float, ticks:Optional[Union[int,float]]) -> None:
        pass

    def log_econ(self,
            ticks:float,
            inventory:npt.NDArray[np.float64],
            balance:npt.NDArray[np.float64],
            buy_prices:npt.NDArray[np.float64],
            buy_budget:npt.NDArray[np.float64],
            sell_prices:npt.NDArray[np.float64],
            max_buy_prices:npt.NDArray[np.float64],
            min_sell_prices:npt.NDArray[np.float64],
            cannot_buy_ticks:npt.NDArray[np.int64],
            cannot_sell_ticks:npt.NDArray[np.int64],
    ) -> None:
        pass

    def flush(self) -> None:
        pass


class StarfieldLayer:
    def __init__(self, bbox:Tuple[float, float, float, float], zoom:float) -> None:
        self.bbox:Tuple[float, float, float, float] = bbox

        # list of stars: loc=(x,y), size, spectral_class
        self._star_list:List[Tuple[Tuple[float, float], float, int]] = []
        self.num_stars = 0

        # density in stars per m^2
        self.density:float = 0.

        # zoom level in meters per character
        self.zoom:float = zoom

    def add_star(self, loc:Tuple[float, float], size:float, spectral_class:int) -> None:
        self._star_list.append((loc, size, spectral_class))
        self.num_stars += 1
        self.density = self.num_stars / ((self.bbox[2]-self.bbox[0])*(self.bbox[3]-self.bbox[1]))
