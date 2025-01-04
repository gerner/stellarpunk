""" Stellarpunk core data model basic objects

No dependencies on other parts of the datamodel
"""

import io
import enum
import abc
import uuid
import itertools
import logging
import collections
import weakref
from collections.abc import Iterable
from typing import Optional, Tuple, List, Sequence, Dict, Any, Collection, Union, Type

import numpy as np
import numpy.typing as npt

from stellarpunk import narrative, util, _version

logger = logging.getLogger(__name__)

def stellarpunk_version() -> str:
    return _version.version

OBSERVER_ID_NULL = uuid.UUID(hex="deadbeefdeadbeefdeadbeefdeadbeef")
class Observer(abc.ABC):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self._observings:weakref.WeakSet[Observable] = weakref.WeakSet()

    @property
    @abc.abstractmethod
    def observer_id(self) -> uuid.UUID:
        """ identifies where this observer is coming from

        Does not have to be unique, this is used for debugging to find a source
        for this observer. Could be an entity_id, even if this Observer is not
        that entity, as long as knowing that entity_id helps to know how to
        find this Observer.

        Ideally knowing the type of the observer and this observer_id is enough
        to uniquely identify the observer, but that's not strictly
        necessary."""
        ...

    @property
    def observings(self) -> Iterable["Observable"]:
        return self._observings

    def mark_observing(self, observed:"Observable") -> None:
        self._observings.add(observed)

    def unmark_observing(self, observed:"Observable") -> None:
        self._observings.remove(observed)

class Observable[T:Observer](abc.ABC):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self._observers:weakref.WeakSet[T] = weakref.WeakSet()

    @property
    @abc.abstractmethod
    def observable_id(self) -> uuid.UUID:
        """ uniquely identifies this observable. e.g. entity_id """
        ...

    @property
    def observers(self) -> Collection[T]:
        return self._observers

    def _observe(self, observer:T) -> None:
        pass

    def _unobserve(self, observer:T) -> None:
        pass

    def observe(self, observer:T) -> None:
        self._observers.add(observer)
        observer.mark_observing(self)

    def unobserve(self, observer:T) -> None:
        # allow double unobserve calls. this might happen because, e.g.
        # Entity.destroy removes observers and the caller might then have
        # further cleanup that removes the observer
        if observer in self._observers:
            self._observers.remove(observer)
            observer.unmark_observing(self)

    def clear_observers(self) -> None:
        for observer in self._observers.copy():
            self.unobserve(observer)

class EntityRegistry(abc.ABC):
    @abc.abstractmethod
    def register_entity(self, entity: "Entity") -> narrative.EventContext: ...

    @abc.abstractmethod
    def destroy_entity(self, entity:"Entity") -> None: ...

    @abc.abstractmethod
    def unregister_entity(self, entity: "Entity") -> None: ...
    @abc.abstractmethod
    def contains_entity(self, entity_id:uuid.UUID) -> bool: ...
    @abc.abstractmethod
    def get_entity[T:"Entity"](self, entity_id:uuid.UUID, klass:Type[T]) -> "Entity": ...
    @abc.abstractmethod
    def now(self) -> float: ...

class Entity(abc.ABC):
    id_prefix = "ENT"

    def __init__(self, entity_registry: EntityRegistry, created_at:Optional[float]=None, name:Optional[str]=None, entity_id:Optional[uuid.UUID]=None, description:Optional[str]=None)->None:
        self.entity_id = entity_id or uuid.uuid4()
        self._entity_id_short_int = util.uuid_to_u64(self.entity_id)

        if name is None:
            name = f'{self.__class__} {str(self.entity_id)}'
        self.name = name

        self.description = description or name
        self.created_at:float = created_at or entity_registry.now()

        self.entity_registry = entity_registry
        self.context = self.entity_registry.register_entity(self)

    def destroy(self) -> None:
        self.entity_registry.unregister_entity(self)
        #import gc
        #referrers = gc.get_referrers(self)
        #if len(referrers) > 0:
        #    raise Exception("ohnoes")

    def short_id(self) -> str:
        """ first 32 bits as hex """
        return f'{self.id_prefix}-{self.entity_id.hex[:8]}'

    def short_id_int(self) -> int:
        return self._entity_id_short_int

    def __str__(self) -> str:
        return f'{self.short_id()}'

    def sanity_check(self) -> None:
        pass


class Sprite:
    """ A "sprite" from a text file that can be drawn in text """

    @staticmethod
    def load_sprites(data:str, sprite_size:Tuple[int, int], sprite_namespace:str) -> List["Sprite"]:
        sprites = []
        sheet = data.split("\n")
        offset_limit = (len(sheet[0])//sprite_size[0], len(sheet)//sprite_size[1])
        sprite_count = 0
        for offset_x, offset_y in itertools.product(range(offset_limit[0]), range(offset_limit[1])):
            sprites.append(Sprite(
                f'{sprite_namespace}.{sprite_count}',
                [
                    x[offset_x*sprite_size[0]:offset_x*sprite_size[0]+sprite_size[0]] for x in sheet[offset_y*sprite_size[1]:offset_y*sprite_size[1]+sprite_size[1]]
                ]
            ))
            sprite_count += 1

        return sprites

    def __init__(self, sprite_id:str, text:Sequence[str], attr:Optional[Dict[Tuple[int,int], Tuple[int, int]]]=None) -> None:
        self.sprite_id = sprite_id
        self.text = text
        self.attr = attr or {}
        self.height = len(text)
        if len(text) > 0:
            self.width = len(text[0])
        else:
            self.width = 0


class EconAgent(Entity, abc.ABC):
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

#TODO: should orders be entities? perhaps some kind of scheduled or action entity
class AbstractOrder(abc.ABC):
    def __init__(self, order_id:Optional[uuid.UUID]=None) -> None:
        if order_id:
            self.order_id = order_id
        else:
            self.order_id = uuid.uuid4()

        self.parent_order:Optional[AbstractOrder] = None
        self.child_orders:collections.deque[AbstractOrder] = collections.deque()

    @abc.abstractmethod
    def to_history(self) -> dict: ...
    @abc.abstractmethod
    def estimate_eta(self) -> float: ...
    @abc.abstractmethod
    def begin_order(self) -> None: ...
    @abc.abstractmethod
    def cancel_order(self) -> None: ...
    @abc.abstractmethod
    def is_complete(self) -> bool: ...
    @abc.abstractmethod
    def pause(self) -> None: ...
    @abc.abstractmethod
    def resume(self) -> None: ...
    @abc.abstractmethod
    def register(self) -> None: ...
    @abc.abstractmethod
    def unregister(self) -> None: ...
    @abc.abstractmethod
    def base_act(self, dt:float) -> None: ...
    @abc.abstractmethod
    def sanity_check(self, order_id:uuid.UUID) -> None: ...

#TODO: should effects be entities? perhaps some kind of scheduled or action entity
class AbstractEffect(abc.ABC):
    def __init__(self, effect_id:Optional[uuid.UUID]=None) -> None:
        if effect_id:
            self.effect_id=effect_id
        else:
            self.effect_id = uuid.uuid4()

    @abc.abstractmethod
    def act(self, dt:float) -> None: ...
    @abc.abstractmethod
    def sanity_check(self, effect_id:uuid.UUID) -> None: ...
    @abc.abstractmethod
    def register(self) -> None: ...
    @abc.abstractmethod
    def unregister(self) -> None: ...
    @abc.abstractmethod
    def begin_effect(self) -> None: ...
    @abc.abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]: ...
