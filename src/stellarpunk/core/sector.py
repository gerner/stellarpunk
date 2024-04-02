""" Sector, containing space filled with SectorEntity objects """

import collections
import uuid
import abc
from typing import List, Any, Dict, Deque, Tuple, Iterator, Union, Optional, MutableMapping, Set

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from .base import Entity
from .sector_entity import SectorEntity, Planet, Station, Asteroid, TravelGate
from .ship import Ship
from .order import Effect

class CollisionObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def collision(self, entity:SectorEntity, other:SectorEntity, impulse:Tuple[float, float], ke:float) -> None: ...

class AbstractSensorManager:
    def __init__(self, sector:"Sector"):
        self.sector = sector

    def compute_effective_profile(self, ship:SectorEntity) -> float:
        return 100.

    def compute_target_profile(self, target:SectorEntity, detector:SectorEntity) -> float:
        return 100.

    def compute_sensor_threshold(self, ship:SectorEntity) -> float:
        return 100.

    def detected(self, target:SectorEntity, detector:SectorEntity) -> bool:
        return True

    def spatial_query(self, detector:SectorEntity, bbox:Tuple[float, float, float, float]) -> Iterator[SectorEntity]:
        return self.sector.spatial_query(bbox)

class Sector(Entity):
    """ A region of space containing resources, stations, ships. """

    id_prefix = "SEC"

    def __init__(self, loc:npt.NDArray[np.float64], radius:float, space:cymunk.Space, *args: Any, **kwargs: Any)->None:
        super().__init__(*args, **kwargs)

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

        self.weather_factor = 1.

        self.sensor_manager:AbstractSensorManager = AbstractSensorManager(self)

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
