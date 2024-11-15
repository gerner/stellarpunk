""" Sector, containing space filled with SectorEntity objects """

import logging
import collections
import uuid
import abc
import enum
import functools
from typing import List, Any, Dict, Deque, Tuple, Iterator, Union, Optional, MutableMapping, Set

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore
import rtree.index # type: ignore

from stellarpunk import util
from .base import Entity
from .sector_entity import SectorEntity, Planet, Station, Asteroid, TravelGate
from .ship import Ship
from .order import Effect

class CollisionObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def collision(self, entity:SectorEntity, other:SectorEntity, impulse:Tuple[float, float], ke:float) -> None: ...

class SensorIdentity:
    def __init__(self, entity:SectorEntity):
        self.object_type = entity.object_type
        self.id_prefix = entity.id_prefix
        self.entity_id = entity.entity_id
        self.short_id = entity.short_id()
        self.radius = entity.radius
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
    def is_detector_active(self) -> bool:
        ...
    @abc.abstractmethod
    def update(self, notify_target:bool=True) -> bool:
        """ Update the image if possible under current sensor conditions

        return true if we're able to detect the target. """
        ...

    @abc.abstractmethod
    def detector_short_id(self) -> str: ...
    @property
    @abc.abstractmethod
    def detector_entity_id(self) -> uuid.UUID: ...

    @abc.abstractmethod
    def copy(self, detector:SectorEntity) -> "AbstractSensorImage": ...

class AbstractSensorSettings:
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
    def __init__(self, sector:"Sector"):
        self.sector = sector

    def compute_effective_profile(self, ship:SectorEntity) -> float:
        return 100.

    def compute_target_profile(self, target:SectorEntity, detector:SectorEntity) -> float:
        return 100.

    def compute_sensor_threshold(self, ship:SectorEntity) -> float:
        return ship.sensor_settings.effective_threshold()

    def detected(self, target:SectorEntity, detector:SectorEntity) -> bool:
        return True

    @abc.abstractmethod
    def spatial_query(self, detector:SectorEntity, bbox:Tuple[float, float, float, float]) -> Iterator[SectorEntity]: ...

    @abc.abstractmethod
    def spatial_point(self, detector:SectorEntity, point:Union[Tuple[float, float], npt.NDArray[np.float64]], max_dist:Optional[float]=None) -> Iterator[SectorEntity]: ...

    @abc.abstractmethod
    def target(self, target:SectorEntity, detector:SectorEntity, notify_target:bool=True) -> AbstractSensorImage: ...
    @abc.abstractmethod
    def sensor_ranges(self, ship:SectorEntity) -> Tuple[float, float, float]: ...
    @abc.abstractmethod
    def profile_ranges(self, ship:SectorEntity) -> Tuple[float, float]: ...

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
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.loc[0]-self.radius, self.loc[1]-self.radius, self.loc[0]+self.radius, self.loc[1]+self.radius)

class SectorWeather:
    """ Represents the effective sector weather for a specific point. """
    def __init__(self) -> None:
        self.sensor_factor = 1.0

    def add(self, region:SectorWeatherRegion) -> None:
        self.sensor_factor *= region.sensor_factor


class Sector(Entity):
    """ A region of space containing resources, stations, ships. """

    id_prefix = "SEC"

    def __init__(self, loc:npt.NDArray[np.float64], radius:float, space:cymunk.Space, *args: Any, **kwargs: Any)->None:
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(util.fullname(self))

        # sector's position in the universe
        self.loc = loc

        # one standard deviation
        self.radius = radius

        self.planets:List[Planet] = []
        self.stations:List[Station] = []
        self.ships:List[Ship] = []
        self.asteroids:Dict[int, List[Asteroid]] = collections.defaultdict(list)

        # id -> entity for all entities in the sector
        self.entities:Dict[uuid.UUID, SectorEntity] = {}

        # physics space for this sector
        # we don't manage this, just have a pointer to it
        # we do rely on this to provide a spatial index of the sector
        self.space:cymunk.Space = space

        self._effects: Deque[Effect] = collections.deque()

        self.collision_observers: MutableMapping[uuid.UUID, Set[CollisionObserver]] = collections.defaultdict(set)

        self._weather_index = rtree.index.Index()
        self._weathers:MutableMapping[int, SectorWeatherRegion] = {}

        self.sensor_manager:AbstractSensorManager = None # type: ignore

    def spatial_query(self, bbox:Tuple[float, float, float, float]) -> Iterator[SectorEntity]:
        for hit in self.space.bb_query(cymunk.BB(*bbox)):
            yield hit.body.data

    def spatial_point(self, point:Union[Tuple[float, float], npt.NDArray[np.float64]], max_dist:Optional[float]=None) -> Iterator[SectorEntity]:
        #TODO: honor mask
        if not max_dist:
            max_dist = np.inf
        for hit in self.space.nearest_point_query(cymunk.vec2d.Vec2d(point[0], point[1]), max_dist):
            yield hit.body.data

    def is_occupied(self, x:float, y:float, eps:float=1e1) -> bool:
        return any(True for _ in self.spatial_query((x-eps, y-eps, x+eps, y+eps)))

    def region_query(self, bbox:Tuple[float, float, float, float]) -> Iterator[SectorWeatherRegion]:
        for hit in self._weather_index.intersection(bbox):
            yield self._weathers[hit]

    def add_region(self, region:SectorWeatherRegion) -> int:
        region.idx = len(self._weather_index)
        self._weathers[region.idx] = region
        self._weather_index.insert(region.idx, region.bbox)
        return region.idx

    @functools.lru_cache(maxsize=4096)
    def _weather_cached(self, loc:Tuple[float, float]) -> SectorWeather:
        """ Caching computation of weather

        This computation is expensive and doesn't change (assuming weather is
        static). Depends on loc being quantized so we get some locality. """
        weather = SectorWeather()
        for idx in self._weather_index.intersection((loc[0], loc[1], loc[0], loc[1])):
            region = self._weathers[idx]
            if util.distance(np.array(loc), region.loc) < region.radius:
                weather.add(self._weathers[idx])
        return weather

    def weather(self, loc:Union[Tuple[float, float], npt.NDArray[np.float64]]) -> SectorWeather:
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

    def add_effect(self, effect:Effect) -> None:
        self._effects.append(effect)
        effect.begin_effect()

    def remove_effect(self, effect:Effect) -> None:
        self._effects.remove(effect)

    def add_entity(self, entity:SectorEntity) -> None:
        #TODO: worry about collisions at location?

        if isinstance(entity, Planet):
            self.planets.append(entity)
        elif isinstance(entity, Station):
            self.stations.append(entity)
        elif isinstance(entity, Ship):
            self.ships.append(entity)
        elif isinstance(entity, Asteroid):
            self.asteroids[entity.resource].append(entity)
        else:
            #isinstance(entity, TravelGate):
            #isinstance(entity, Missile):
            #raise ValueError(f'unknown entity type {entity.__class__}')
            pass

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

        if isinstance(entity, Planet):
            self.planets.remove(entity)
        elif isinstance(entity, Station):
            self.stations.remove(entity)
        elif isinstance(entity, Ship):
            self.ships.remove(entity)
        elif isinstance(entity, Asteroid):
            self.asteroids[entity.resource].remove(entity)
        else:
            #isinstance(entity, TravelGate):
            #isinstance(entity, Missile):
            #raise ValueError(f'unknown entity type {entity.__class__}')
            pass

        if entity.phys.is_static:
            self.space.remove(entity.phys_shape)
        else:
            self.space.remove(entity.phys, entity.phys_shape)
        entity.sector = None
        del self.entities[entity.entity_id]
