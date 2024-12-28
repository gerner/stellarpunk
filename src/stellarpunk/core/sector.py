""" Sector, containing space filled with SectorEntity objects """

import logging
import collections
import uuid
import abc
import enum
import functools
import gzip
import json
import weakref
from typing import Optional, Union, Any, TextIO, Type
from collections.abc import Iterable, Iterator, MutableMapping, Collection, Sequence, Generator

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore
import rtree.index # type: ignore

from stellarpunk import util
from . import base

SECTOR_ENTITY_COLLISION_TYPE = 0

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
            order_hist:Optional[dict]=None
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

    def to_json(self) -> dict[str, Any]:
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

def write_history_to_file(entity:Union["Sector", "SectorEntity"], f:Union[str, TextIO], mode:str="w", now:float=-np.inf) -> None:
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

class SectorEntityObserver(base.Observer):
    def entity_migrated(self, entity:"SectorEntity", from_sector:"Sector", to_sector:"Sector") -> None:
        pass

    def entity_destroyed(self, entity:"SectorEntity") -> None:
        pass

    def entity_targeted(self, entity:"SectorEntity", threat:"SectorEntity") -> None:
        pass

class SectorEntity(base.Observable[SectorEntityObserver], base.Entity):
    """ An entity in space in a sector. """

    def __init__(self, loc:npt.NDArray[np.float64], phys: cymunk.Body, num_products:int, sensor_settings:"AbstractSensorSettings", *args:Any, history_length:int=60*60, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.sector:Optional["Sector"] = None

        # some physical properties (in SI units)
        self.mass = 0.
        self.moment = 0.
        #TODO: are all entities just circles?
        self.radius = 0.
        self.cargo_capacity = 5e2

        self.cargo:npt.NDArray[np.float64] = np.zeros((num_products,))

        assert(not np.isnan(loc[0]))
        assert(not np.isnan(loc[1]))

        phys.position = (loc[0], loc[1])

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = phys
        self.phys_shape:Any = None

        self.history: collections.deque[HistoryEntry] = collections.deque(maxlen=history_length)

        self.sensor_settings=sensor_settings

    # base.Observable
    @property
    def observable_id(self) -> uuid.UUID:
        return self.entity_id


    def migrate(self, to_sector:"Sector") -> None:
        if self.sector is None:
            raise Exception("cannot migrate if from sector is None")
        from_sector = self.sector
        if self.sector is not None:
            self.sector.remove_entity(self)
        to_sector.add_entity(self)
        for o in self._observers.copy():
            o.entity_migrated(self, from_sector, to_sector)

        self._migrate(to_sector)

    def _migrate(self, to_sector:"Sector") -> None:
        pass

    def destroy(self) -> None:
        for o in self._observers.copy():
            o.entity_destroyed(self)
        self._observers.clear()

        self._destroy()
        self.phys.data = None
        super().destroy()

    def _destroy(self) -> None:
        pass

    def target(self, threat:"SectorEntity") -> None:
        for o in self._observers.copy():
            o.entity_targeted(self, threat)

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

class CollisionObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def collision(self, entity:SectorEntity, other:SectorEntity, impulse:tuple[float, float], ke:float) -> None: ...

class SensorIdentity:
    def __init__(self, entity:Optional[SectorEntity]=None, object_type:Optional[Type[SectorEntity]]=None, id_prefix:Optional[str]=None, entity_id:Optional[uuid.UUID]=None, short_id:Optional[str]=None, radius:Optional[float]=None):
        if entity:
            self.object_type:Type[SectorEntity]=type(entity)
            self.id_prefix = entity.id_prefix
            self.entity_id = entity.entity_id
            self.short_id = entity.short_id()
            self.radius = entity.radius
        else:
            assert(object_type)
            assert(id_prefix)
            assert(entity_id)
            assert(short_id)
            assert(radius)
            self.object_type = object_type
            self.id_prefix = id_prefix
            self.entity_id = entity_id
            self.short_id = short_id
            self.radius = radius
        # must be updated externally
        self.angle = 0.0

class SensorImageInactiveReason(enum.IntEnum):
    OTHER = enum.auto()
    DESTROYED = enum.auto()
    MIGRATED = enum.auto()

class AbstractSensorImage:
    """ A sensor contact which might be old with predicted attributes

    This lets logic interact with a target through sensors, without needing to
    directly hang on to the target object. This image predicts the target's 
    position and velocity using latest sensor readings.
    """
    @property
    @abc.abstractmethod
    def age(self) -> float:
        """ How long ago was this image updated in seconds """
        ...
    @property
    @abc.abstractmethod
    def loc(self) -> npt.NDArray[np.float64]:
        """ Predicted location of the image """
        ...
    @property
    @abc.abstractmethod
    def velocity(self) -> npt.NDArray[np.float64]:
        """ Predicted velocity of the image """
        ...
    @property
    @abc.abstractmethod
    def acceleration(self) -> npt.NDArray[np.float64]:
        """ Predicted acceleration of the image """
        ...
    @property
    @abc.abstractmethod
    def profile(self) -> float: ...
    @property
    @abc.abstractmethod
    def fidelity(self) -> float: ...
    @property
    @abc.abstractmethod
    def identified(self) -> bool: ...
    @property
    @abc.abstractmethod
    def identity(self) -> SensorIdentity: ...
    @property
    @abc.abstractmethod
    def transponder(self) -> bool: ...
    @abc.abstractmethod
    def is_active(self) -> bool:
        """ False iff we detected target destroyed or leaving the sector """
        ...
    @property
    @abc.abstractmethod
    def inactive_reason(self) -> SensorImageInactiveReason: ...
    @abc.abstractmethod
    def update(self, notify_target:bool=True) -> bool:
        """ Update the image if possible under current sensor conditions

        return true if we're able to detect the target. """
        ...

    @abc.abstractmethod
    def set_sensor_manager(self, sensor_manager:"AbstractSensorManager") -> None: ...

    @abc.abstractmethod
    def copy(self, detector:SectorEntity) -> "AbstractSensorImage": ...

class AbstractSensorSettings:
    @abc.abstractmethod
    def register_image(self, image:AbstractSensorImage) -> None: ...
    @abc.abstractmethod
    def unregister_image(self, image:AbstractSensorImage) -> None: ...
    @abc.abstractmethod
    def has_image(self, target_id:uuid.UUID) -> bool: ...
    @abc.abstractmethod
    def get_image(self, target_id:uuid.UUID) -> AbstractSensorImage: ...
    @property
    @abc.abstractmethod
    def images(self) -> Collection[AbstractSensorImage]: ...

    @property
    @abc.abstractmethod
    def max_sensor_power(self) -> float: ...
    @property
    @abc.abstractmethod
    def sensor_power(self) -> float: ...
    @property
    @abc.abstractmethod
    def transponder(self) -> bool: ...
    @abc.abstractmethod
    def effective_sensor_power(self) -> float: ...
    @abc.abstractmethod
    def effective_transponder(self) -> float: ...
    @abc.abstractmethod
    def effective_thrust(self) -> float: ...
    @abc.abstractmethod
    def effective_threshold(self, sensor_power:Optional[float]=None) -> float: ...
    @property
    @abc.abstractmethod
    def max_threshold(self) -> float: ...
    @abc.abstractmethod
    def set_sensors(self, ratio:float) -> None: ...
    @abc.abstractmethod
    def set_transponder(self, on:bool) -> None: ...
    @abc.abstractmethod
    def set_thrust(self, thrust:float) -> None: ...
    @property
    @abc.abstractmethod
    def thrust_seconds(self) -> float: ...
    @property
    @abc.abstractmethod
    def ignore_bias(self) -> bool: ...

class AbstractSensorManager:
    @abc.abstractmethod
    def compute_effective_profile(self, ship:SectorEntity) -> float: ...

    @abc.abstractmethod
    def compute_effective_threshold(self, ship:SectorEntity) -> float: ...


    def compute_target_profile(self, target:SectorEntity, detector:SectorEntity) -> float:
        return 100.

    def compute_sensor_threshold(self, ship:SectorEntity) -> float:
        return ship.sensor_settings.effective_threshold()

    def detected(self, target:SectorEntity, detector:SectorEntity) -> bool:
        return True

    @abc.abstractmethod
    def spatial_query(self, detector:SectorEntity, bbox:tuple[float, float, float, float]) -> Iterator[SectorEntity]: ...

    @abc.abstractmethod
    def spatial_point(self, detector:SectorEntity, point:Union[tuple[float, float], npt.NDArray[np.float64]], max_dist:Optional[float]=None) -> Iterator[SectorEntity]: ...

    @abc.abstractmethod
    def target(self, target:SectorEntity, detector:SectorEntity, notify_target:bool=True) -> AbstractSensorImage: ...
    @abc.abstractmethod
    def sensor_ranges(self, ship:SectorEntity) -> tuple[float, float, float]: ...
    @abc.abstractmethod
    def profile_ranges(self, ship:SectorEntity) -> tuple[float, float]: ...

    @abc.abstractmethod
    def compute_thrust_for_profile(self, ship:SectorEntity, distance_sq:float, profile:float) -> float: ...
    @abc.abstractmethod
    def compute_thrust_for_sensor_power(self, ship:SectorEntity, distance_sq:float, sensor_power:float) -> float: ...


class SectorWeatherRegion:
    """ Models weather effects in a sector.

    These are circular disk shaped regions that have various impacts.
    Overlapping weather regions combine into an effective weather for every
    point in the region. """

    def __init__(self, loc:npt.NDArray[np.float64], radius:float, sensor_factor:float) -> None:
        self.idx = -1
        self.loc = loc
        self.radius = radius

        # other weather properties (e.g. sensor factor)
        self.sensor_factor = sensor_factor

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.loc[0]-self.radius, self.loc[1]-self.radius, self.loc[0]+self.radius, self.loc[1]+self.radius)

class SectorWeather:
    """ Represents the effective sector weather for a specific point. """
    def __init__(self) -> None:
        self.sensor_factor = 1.0

    def add(self, region:SectorWeatherRegion) -> None:
        self.sensor_factor *= region.sensor_factor


class Sector(base.Entity):
    """ A region of space containing resources, stations, ships. """

    id_prefix = "SEC"

    def __init__(self, loc:npt.NDArray[np.float64], radius:float, space:cymunk.Space, *args: Any, culture:str, **kwargs: Any)->None:
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(util.fullname(self))

        # sector's position in the universe
        self.loc = loc

        # one standard deviation
        self.radius = radius

        # a "culture" for the sector which helps with consistent naming
        self.culture = culture

        # id -> entity for all entities in the sector
        self.entities:dict[uuid.UUID, SectorEntity] = {}

        # physics space for this sector
        # we don't manage this, just have a pointer to it
        # we do rely on this to provide a spatial index of the sector
        self.space:cymunk.Space = space

        self._effects: collections.deque[base.AbstractEffect] = collections.deque()

        self.collision_observers: MutableMapping[uuid.UUID, set[CollisionObserver]] = collections.defaultdict(set)

        self._weather_index = rtree.index.Index()
        self._weathers:MutableMapping[int, SectorWeatherRegion] = {}

        self.sensor_manager:AbstractSensorManager = None # type: ignore

    def spatial_query(self, bbox:tuple[float, float, float, float]) -> Iterator[SectorEntity]:
        for hit in self.space.bb_query(cymunk.BB(*bbox)):
            if hit.collision_type != SECTOR_ENTITY_COLLISION_TYPE:
                continue
            yield hit.body.data

    def spatial_point(self, point:Union[tuple[float, float], npt.NDArray[np.float64]], max_dist:Optional[float]=None) -> Iterator[SectorEntity]:
        #TODO: honor mask
        if not max_dist:
            max_dist = np.inf
        for hit in self.space.nearest_point_query(cymunk.vec2d.Vec2d(point[0], point[1]), max_dist):
            if hit.collision_type != SECTOR_ENTITY_COLLISION_TYPE:
                continue
            yield hit.body.data

    def is_occupied(self, x:float, y:float, eps:float=1e1) -> bool:
        return any(True for _ in self.spatial_query((x-eps, y-eps, x+eps, y+eps)))

    def region_query(self, bbox:tuple[float, float, float, float]) -> Iterator[SectorWeatherRegion]:
        for hit in self._weather_index.intersection(bbox):
            yield self._weathers[hit]

    def add_region(self, region:SectorWeatherRegion) -> int:
        region.idx = len(self._weather_index)
        self._weathers[region.idx] = region
        self._weather_index.insert(region.idx, region.bbox)
        return region.idx

    @functools.lru_cache(maxsize=4096)
    def _weather_cached(self, loc:tuple[float, float]) -> SectorWeather:
        """ Caching computation of weather

        This computation is expensive and doesn't change (assuming weather is
        static). Depends on loc being quantized so we get some locality. """
        weather = SectorWeather()
        for idx in self._weather_index.intersection((loc[0], loc[1], loc[0], loc[1])):
            region = self._weathers[idx]
            if util.distance(np.array(loc), region.loc) < region.radius:
                weather.add(self._weathers[idx])
        return weather

    def weather(self, loc:Union[tuple[float, float], npt.NDArray[np.float64]]) -> SectorWeather:
        # quantize loc so we can cache it
        quantized_loc = (loc[0] // 100.0 * 100.0, loc[1] // 100.0 * 100.0)
        return self._weather_cached(quantized_loc)

    def register_collision_observer(self, entity_id:uuid.UUID, observer:CollisionObserver) -> None:
        self.collision_observers[entity_id].add(observer)

    def unregister_collision_observer(self, entity_id:uuid.UUID, observer:CollisionObserver) -> None:
        try:
            self.collision_observers[entity_id].remove(observer)
        except KeyError:
            # allow double unregister (e.g. the observe wants to unregister
            # after we've destroyed the thing it observes)
            pass

    def add_effect(self, effect:base.AbstractEffect) -> None:
        effect.register()
        self._effects.append(effect)
        effect.begin_effect()

    def remove_effect(self, effect:base.AbstractEffect) -> None:
        self._effects.remove(effect)
        effect.unregister()

    def current_effects(self) -> Iterable[base.AbstractEffect]:
        return self._effects

    def entities_by_type[T:SectorEntity](self, klass:Type[T]) -> Generator[T, None, None]:
        for v in self.entities.values():
            if isinstance(v, klass):
                yield v

    def add_entity(self, entity:SectorEntity) -> None:

        collision_entity = next(self.spatial_point(entity.loc, entity.radius), None)
        if collision_entity:
            raise ValueError(f'tried to place {entity} on top of {collision_entity}')

        if entity.phys.is_static:
            self.space.add(entity.phys_shape)
        else:
            self.space.add(entity.phys, entity.phys_shape)
        entity.sector = self
        self.entities[entity.entity_id] = entity

    def remove_entity(self, entity:SectorEntity) -> None:
        self.logger.debug(f'removing {entity} {entity.sensor_settings.thrust_seconds=}')

        if entity.entity_id not in self.entities:
            raise ValueError(f'entity {entity.entity_id} not in this sector')

        if entity.entity_id in self.collision_observers:
            del self.collision_observers[entity.entity_id]

        if entity.phys.is_static:
            self.space.remove(entity.phys_shape)
        else:
            self.space.remove(entity.phys, entity.phys_shape)
        entity.sector = None
        del self.entities[entity.entity_id]
