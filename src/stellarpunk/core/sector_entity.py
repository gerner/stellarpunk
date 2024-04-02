""" Sector Entities that live in a Sector, have physics, etc. """

import enum
import uuid
import collections
import gzip
import json
from typing import Optional, Dict, Mapping, Any, Deque, Sequence, Union, TextIO, Iterable, TYPE_CHECKING, Set

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util
from .base import Entity, Sprite, Asset

if TYPE_CHECKING:
    from . import sector, character

class ObjectType(enum.IntEnum):
    OTHER = enum.auto()
    SHIP = enum.auto()
    STATION = enum.auto()
    PLANET = enum.auto()
    ASTEROID = enum.auto()
    TRAVEL_GATE = enum.auto()
    MISSILE = enum.auto()


#class ObjectFlag(enum.IntFlag):
#    # note: with pymunk we get up to 32 of these (depending on the c-type?)
#    SHIP = enum.auto()
#    STATION = enum.auto()
#    PLANET = enum.auto()
#    ASTEROID = enum.auto()
#    GATE = enum.auto()
#    MISSILE = enum.auto()


class HistoryEntry:
    def __init__(
            self,
            prefix:str,
            entity_id:uuid.UUID,
            ts:float,
            loc:tuple,
            radius:float,
            angle:float,
            velocity:tuple,
            angular_velocity:float,
            force:tuple,
            torque:float,
            order_hist:Optional[Dict]=None
    ) -> None:
        self.entity_id = entity_id
        self.ts = ts

        self.order_hist = order_hist

        self.loc = loc
        self.radius = radius
        self.angle = angle
        self.prefix = prefix

        self.velocity = velocity
        self.angular_velocity = angular_velocity

        self.force = force
        self.torque = torque

    def to_json(self) -> Mapping[str, Any]:
        return {
            "p": self.prefix,
            "eid": str(self.entity_id),
            "ts": self.ts,
            "loc": self.loc,
            "r": self.radius,
            "a": self.angle,
            "v": self.velocity,
            "av": self.angular_velocity,
            "f": self.force,
            "t": self.torque,
            "o": self.order_hist,
        }

class SectorEntityObserver:
    def entity_migrated(self, entity:"SectorEntity", from_sector:"sector.Sector", to_sector:"sector.Sector") -> None:
        pass

    def entity_destroyed(self, entity:"SectorEntity") -> None:
        pass

class SectorEntity(Entity):
    """ An entity in space in a sector. """

    object_type = ObjectType.OTHER

    def __init__(self, loc:npt.NDArray[np.float64], phys: cymunk.Body, num_products:int, *args:Any, history_length:int=60*60, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.sector:Optional["sector.Sector"] = None

        # some physical properties (in SI units)
        self.mass = 0.
        self.moment = 0.

        phys.position = (loc[0], loc[1])

        self.cargo_capacity = 5e2

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = phys
        self.phys_shape:Any = None
        #TODO: are all entities just circles?
        self.radius = 0.

        self.history: Deque[HistoryEntry] = collections.deque(maxlen=history_length)

        self.cargo:npt.NDArray[np.float64] = np.zeros((num_products,))

        # who is responsible for this entity?
        self.captain: Optional["character.Character"] = None

        self.observers:Set[SectorEntityObserver] = set()

        self.sensor_power = 0.
        self.transponder_on = False

    def observe(self, observer:SectorEntityObserver) -> None:
        self.observers.add(observer)

    def unobserve(self, observer:SectorEntityObserver) -> None:
        try:
            self.observers.remove(observer)
        except KeyError:
            pass

    def migrate(self, to_sector:"sector.Sector") -> None:
        if self.sector is None:
            raise Exception("cannot migrate if from sector is None")
        from_sector = self.sector
        if self.sector is not None:
            self.sector.remove_entity(self)
        to_sector.add_entity(self)
        for o in self.observers.copy():
            o.entity_migrated(self, from_sector, to_sector)

        self._migrate(to_sector)

    def _migrate(self, to_sector:"sector.Sector") -> None:
        pass

    def destroy(self) -> None:
        for o in self.observers.copy():
            o.entity_destroyed(self)
        self.observers.clear()

        self._destroy()
        self.phys.data = None
        super().destroy()

    def _destroy(self) -> None:
        pass

    @property
    def loc(self) -> npt.NDArray[np.float64]: return np.array(self.phys.position)
    @property
    def velocity(self) -> npt.NDArray[np.float64]: return np.array(self.phys.velocity)
    @property
    def speed(self) -> float: return self.phys.velocity.length
    @property
    def angle(self) -> float: return self.phys.angle
    @property
    def angular_velocity(self) -> float: return self.phys.angular_velocity

    def distance_to(self, other:"SectorEntity") -> float:
        if other.sector != self.sector:
            raise ValueError(f'other in sector {other.sector} but we are in sector {self.sector}')
        return util.distance(self.loc, other.loc) - self.radius - other.radius

    def cargo_full(self) -> bool:
        return np.sum(self.cargo) == self.cargo_capacity

    def get_history(self) -> Sequence[HistoryEntry]:

        return (HistoryEntry(
                self.id_prefix,
                self.entity_id, 0,
                tuple(self.phys.position), self.radius, self.angle,
                tuple(self.phys.velocity), self.angular_velocity,
                (0.,0.), 0,
        ),)
        return self.history

    def to_history(self, timestamp:float) -> HistoryEntry:
        return HistoryEntry(
                self.id_prefix,
                self.entity_id, timestamp,
                tuple(self.phys.position), self.radius, self.angle,
                tuple(self.phys.velocity), self.angular_velocity,
                (0.,0.), 0,
        )
    def address_str(self) -> str:
        if self.sector:
            return f'{self.short_id()}@{self.sector.short_id()}'
        else:
            return f'{self.short_id()}@None'


class Planet(SectorEntity, Asset):

    id_prefix = "HAB"
    object_type = ObjectType.PLANET

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.population = 0.

        self.sensor_power = 1000.
        self.transponder_on = True


class Station(SectorEntity, Asset):

    id_prefix = "STA"
    object_type = ObjectType.STATION

    def __init__(self, sprite:Sprite, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource: Optional[int] = None
        self.next_batch_time = 0.
        self.next_production_time = 0.
        self.cargo_capacity = 1e5

        self.sprite = sprite

        self.sensor_power = 1000.
        self.transponder_on = True

class Asteroid(SectorEntity):

    id_prefix = "AST"
    object_type = ObjectType.ASTEROID

    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.cargo[self.resource] = amount


class TravelGate(SectorEntity):
    """ Represents a "gate" to another sector """

    id_prefix = "GAT"
    object_type = ObjectType.TRAVEL_GATE

    def __init__(self, destination:"sector.Sector", direction:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.destination = destination
        # radian angle toward the destination
        self.direction:float = direction
        self.direction_vector = np.array(util.polar_to_cartesian(1., direction))


def write_history_to_file(entity:Union["sector.Sector", SectorEntity], f:Union[str, TextIO], mode:str="w", now:float=-np.inf) -> None:
    fout:TextIO
    if isinstance(f, str):
        needs_close = True
        if f.endswith(".gz"):
            fout = gzip.open(f, mode+"t") # type: ignore[assignment]
        else:
            fout = open(f, mode) # type: ignore[assignment]
    else:
        needs_close = False
        fout = f

    entities:Iterable[SectorEntity]
    if isinstance(entity, SectorEntity):
        entities = [entity]
    else:
        entities = entity.entities.values()

    for ent in entities:
        history = ent.get_history()
        for entry in history:
            fout.write(json.dumps(entry.to_json()))
            fout.write("\n")
        if len(history) == 0 or history[-1].ts < now:
            fout.write(json.dumps(ent.to_history(now).to_json()))
            fout.write("\n")
    if needs_close:
        fout.close()
