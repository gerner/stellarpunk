""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

from typing import Tuple, Iterator, Optional, Any, Union, Dict, Mapping, Iterable, Set
import uuid
import logging

import rtree.index
import numpy.typing as npt
import numpy as np

from stellarpunk import core, config, util

logger = logging.getLogger(__name__)

ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False

class SensorImage(core.AbstractSensorImage, core.SectorEntityObserver):
    def __init__(self, target:Optional[core.SectorEntity], ship:core.SectorEntity, sensor_manager:"SensorManager", identity:Optional[core.SensorIdentity]=None) -> None:
        self._sensor_manager = sensor_manager
        self._ship:Optional[core.SectorEntity] = ship
        self._detector_entity_id = ship.entity_id
        self._detector_short_id = ship.short_id()
        self._target:Optional[core.SectorEntity] = target
        if target:
            if identity is None:
                self._identity = core.SensorIdentity(target)
            else:
                self._identity = identity
            target.observe(self)
        else:
            if identity is None:
                raise ValueError("must provide a sensor identity if no target is given")
            self._identity = identity
        ship.observe(self)
        self._last_update = 0.
        self._prior_fidelity = 1.0
        self._last_profile = 0.
        self._loc = ZERO_VECTOR
        self._velocity = ZERO_VECTOR
        self._acceleration = ZERO_VECTOR
        self._is_active = True
        self._inactive_reason = core.SensorImageInactiveReason.OTHER

        # models noise in the sensors
        self._loc_bias_direction = np.array((0.0, 0.0))
        self._loc_bias = np.array((0.0, 0.0))
        self._velocity_bias = np.array((0.0, 0.0))
        self._last_bias_update_ts = -np.inf

        self._identified = False

    def __str__(self) -> str:
        return f'{self._identity.short_id} detected by {self._detector_short_id} {self.age}s old'

    def __hash__(self) -> int:
        return hash((self._identity.entity_id, self._detector_entity_id))

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self._target:
            # might be risky checking if detected on logically destroyed entity
            if self._ship and self._sensor_manager.detected(entity, self._ship):
                self._is_active = False
                self._inactive_reason = core.SensorImageInactiveReason.DESTROYED
            self._target = None
            if self._ship:
                self._ship.unobserve(self)
        elif entity == self._ship:
            if self._target:
                self._target.unobserve(self)
                self._target = None
            self._ship.unobserve(self)
            self._ship = None
        else:
            raise ValueError(f'got entity_destroyed for unexpected entity {entity}')

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if entity == self._target:
            # might be risky checking if detected on logically migrated entity
            if self._ship and self._sensor_manager.detected(entity, self._ship):
                self._is_active = False
                self._inactive_reason = core.SensorImageInactiveReason.MIGRATED
            self._target.unobserve(self)
            self._target = None
        elif entity == self._ship:
            if self._target:
                self._target.unobserve(self)
                self._target = None
            self._ship.unobserve(self)
            self._ship = None
        else:
            raise ValueError(f'got entity_migrated for unexpected entity {entity}')

    @property
    def detector_entity_id(self) -> uuid.UUID:
        return self._detector_entity_id
    def detector_short_id(self) -> str:
        return self._detector_short_id

    @property
    def age(self) -> float:
        return core.Gamestate.gamestate.timestamp - self._last_update

    @property
    def loc(self) -> npt.NDArray[np.float64]:
        assert self._ship
        if self._ship.sensor_settings.ignore_bias:
            return (self._loc) + (self._velocity) * (core.Gamestate.gamestate.timestamp - self._last_update)
        else:
            return (self._loc + self._loc_bias) + (self._velocity + self._velocity_bias) * (core.Gamestate.gamestate.timestamp - self._last_update)

    @property
    def profile(self) -> float:
        return self._last_profile

    @property
    def fidelity(self) -> float:
        assert self._ship
        return self.profile / self._sensor_manager.compute_effective_threshold(self._ship)

    @property
    def identified(self) -> bool:
        return self._identified

    @property
    def identity(self) -> core.SensorIdentity:
        return self._identity

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        assert self._ship
        if self._ship.sensor_settings.ignore_bias:
            return self._velocity
        else:
            return self._velocity + self._velocity_bias

    @property
    def acceleration(self) -> npt.NDArray[np.float64]:
        return self._acceleration

    @property
    def transponder(self) -> bool:
        if self._target:
            return self._target.sensor_settings.transponder
        else:
            return False

    def is_active(self) -> bool:
        return self._is_active

    @property
    def inactive_reason(self) -> core.SensorImageInactiveReason:
        return self._inactive_reason

    def is_detector_active(self) -> bool:
        return self._ship is not None

    def initialize(self) -> None:
        self._initialize_bias()

    def _initialize_bias(self) -> None:
        assert self._target
        assert self._ship

        # choose a direction for the location bias
        # choose an angle and radius offset relative to detector
        distance = util.magnitude(*(self._target.loc - self._ship.loc))
        if distance > 0.0:
            offset_r = core.Gamestate.gamestate.random.uniform(low=-1.0, high=1.0) * config.Settings.sensors.COEFF_BIAS_OFFSET_R * distance
            offset_theta = core.Gamestate.gamestate.random.uniform(low=-1.0, high=1.0) * config.Settings.sensors.COEFF_BIAS_OFFSET_THETA

            target_r, target_theta = util.cartesian_to_polar(*(self._target.loc - self._ship.loc))

            self._loc_bias_direction = np.array(util.polar_to_cartesian(target_r+offset_r, target_theta+offset_theta)) + self._ship.loc - self._target.loc
            self._loc_bias_direction /= util.magnitude(*self._loc_bias_direction)
        self._last_bias_update_ts = -np.inf

    def _update_bias(self) -> None:

        assert self._target
        assert self._ship

        if self._target.object_type == core.ObjectType.PROJECTILE:
            self._last_bias_update_ts = core.Gamestate.gamestate.timestamp
            return

        new_loc_bias = self._loc_bias_direction / self.fidelity * config.Settings.sensors.COEFF_BIAS_LOC

        #TODO: add some extra noise to new_loc_bias?

        time_delta = core.Gamestate.gamestate.timestamp - self._last_bias_update_ts
        if self.fidelity < self._prior_fidelity:
            alpha = 1/(-(1/config.Settings.sensors.COEFF_BIAS_TIME_DECAY_DOWN * time_delta + 1)) + 1
        else:
            alpha = 1/(-(1/config.Settings.sensors.COEFF_BIAS_TIME_DECAY_UP * time_delta + 1)) + 1

        if self._target.object_type == core.ObjectType.MISSILE:
            core.Gamestate.gamestate.breakpoint()

        new_loc_bias = alpha * new_loc_bias + (1-alpha) * self._loc_bias
        #new_velocity_bias = np.array((0.0, 0.0))

        self._loc_bias = new_loc_bias
        #self._velocity_bias = new_velocity_bias
        self._last_bias_update_ts = core.Gamestate.gamestate.timestamp

    def update(self, notify_target:bool=True) -> bool:
        if self._target and self._ship:
            # update sensor reading if possible
            if self._sensor_manager.detected(self._target, self._ship):
                self._prior_fidelity = self.fidelity
                self._identity.angle = self._target.angle
                self._last_profile = self._sensor_manager.compute_target_profile(self._target, self._ship)
                self._update_bias()

                since_last_update = core.Gamestate.gamestate.timestamp - self._last_update
                if since_last_update < config.Settings.sensors.ACCEL_PREDICTION_MAX_SEC:
                    self._acceleration = self._target.velocity - self._velocity
                else:
                    self._acceleration = ZERO_VECTOR

                self._last_update = core.Gamestate.gamestate.timestamp
                self._loc = self._target.loc
                self._velocity = np.array(self._target.velocity)

                if self.fidelity * config.Settings.sensors.COEFF_IDENTIFICATION_FIDELITY > 1.0:
                    #if not self._identified:
                    #    logger.debug(f'{self._ship.short_id()} identified {self._target.short_id()} with fidelity={self.fidelity}')
                    # once identified, always identified
                    self._identified = True

                # let the target know they've been targeted by us
                if notify_target and self._sensor_manager.detected(self._ship, self._target):
                    self._target.target(self._ship)
                return True
            else:
                self._acceleration = ZERO_VECTOR
                return False
        else:
            #TODO: shouldn't we flip _is_active at some point?
            return False

    def copy(self, detector:core.SectorEntity) -> core.AbstractSensorImage:
        """ Retargets the underlying craft with a (potentially) new detector

        This does not reset state, but future updates will be with respect to
        the new detector. """
        identity = self._identity
        image = SensorImage(self._target, detector, self._sensor_manager, identity=self._identity)
        image._last_update = self._last_update
        image._last_profile = self._last_profile
        image._loc = self._loc
        image._velocity = self._velocity
        image._acceleration = self._acceleration
        image._is_active = self._is_active

        image._loc_bias = self._loc_bias
        image._velocity_bias = self._velocity_bias
        image._last_bias_update_ts = self._last_bias_update_ts

        assert image._ship

        return image

class SensorSettings(core.AbstractSensorSettings):
    def __init__(self, max_sensor_power:float=0., sensor_intercept:float=100.0) -> None:
        self._max_sensor_power = max_sensor_power
        self._sensor_intercept = sensor_intercept
        self._sensor_power = 0.
        self._transponder_on = False
        self._thrust = 0.

        # keeps track of history on sensor spikes
        self._last_sensor_power = 0.
        self._last_sensor_power_ts = 0.
        self._last_transponder_ts = -np.inf
        self._last_thrust = 0.
        self._last_thrust_ts = 0.

        self._thrust_seconds = 0.

        self._ignore_bias = False

    @property
    def max_sensor_power(self) -> float:
        return self._max_sensor_power
    @property
    def sensor_power(self) -> float:
        return self._sensor_power
    @property
    def transponder(self) -> bool:
        return self._transponder_on

    @property
    def max_threshold(self) -> float:
        return config.Settings.sensors.COEFF_THRESHOLD / (self._max_sensor_power + config.Settings.sensors.COEFF_THRESHOLD/self._sensor_intercept)

    def effective_sensor_power(self) -> float:
        if self._last_sensor_power == self._sensor_power:
            return self._sensor_power
        return (self._last_sensor_power - self._sensor_power) * config.Settings.sensors.DECAY_SENSORS ** (core.Gamestate.gamestate.timestamp - self._last_sensor_power_ts) + self._sensor_power
    def effective_transponder(self) -> float:
        if self._transponder_on:
            return 1.0
        return config.Settings.sensors.DECAY_TRANSPONDER ** (core.Gamestate.gamestate.timestamp - self._last_transponder_ts)
    def effective_thrust(self) -> float:
        if self._last_thrust == self._thrust:
            return self._thrust
        return (self._last_thrust - self._thrust) * config.Settings.sensors.DECAY_THRUST ** (core.Gamestate.gamestate.timestamp - self._last_thrust_ts) + self._thrust
    def effective_threshold(self, effective_sensor_power:Optional[float]=None) -> float:
        """ computes the sensor threshold accounting for sensor power """
        if effective_sensor_power is None:
            effective_sensor_power = self.effective_sensor_power()
        return config.Settings.sensors.COEFF_THRESHOLD / (effective_sensor_power + config.Settings.sensors.COEFF_THRESHOLD/self._sensor_intercept)

    def set_sensors(self, ratio:float) -> None:
        # keep track of spikes in sensors so impact on profile decays with time
        # sensors decay up and down
        new_level = ratio * self._max_sensor_power
        if new_level == self._sensor_power:
            return
        decayed_power = self.effective_sensor_power()
        self._last_sensor_power = decayed_power
        self._last_sensor_power_ts = core.Gamestate.gamestate.timestamp
        self._sensor_power = new_level
    def set_transponder(self, on:bool) -> None:
        # keep track of transponder so impact on profile decays with time
        if self._transponder_on:
            self._last_transponder_ts = core.Gamestate.gamestate.timestamp
        self._transponder_on = on
    def set_thrust(self, thrust:float) -> None:
        # keep track of spikes in thrust so impact on profile decays with time
        # thrust only decays down, it spikes instantly up
        if thrust == self._thrust:
            return
        decayed_thrust = self.effective_thrust()
        if thrust > decayed_thrust:
            self._last_thrust = thrust
        else:
            self._last_thrust = decayed_thrust
        self._thrust_seconds += self._thrust * (core.Gamestate.gamestate.timestamp - self._last_thrust_ts)
        self._last_thrust_ts = core.Gamestate.gamestate.timestamp
        self._thrust = thrust

    @property
    def thrust_seconds(self) -> float:
        return self._thrust_seconds

    @property
    def ignore_bias(self) -> bool:
        return self._ignore_bias


class SensorManager(core.AbstractSensorManager):
    """ Models sensors for a sector """

    def __init__(self, sector:core.Sector):
        self.sector = sector

    def compute_effective_threshold(self, ship:core.SectorEntity) -> float:
        return ship.sensor_settings.effective_threshold()

    def compute_effective_profile(self, ship:core.SectorEntity) -> float:
        """ computes the profile  accounting for ship and sector factors """
        return (
            config.Settings.sensors.COEFF_MASS * ship.mass +
            config.Settings.sensors.COEFF_RADIUS * ship.radius +
            config.Settings.sensors.COEFF_FORCE * ship.sensor_settings.effective_thrust()**config.Settings.sensors.FORCE_EXPONENT +
            config.Settings.sensors.COEFF_SENSORS * ship.sensor_settings.effective_sensor_power() +
            config.Settings.sensors.COEFF_TRANSPONDER * ship.sensor_settings.effective_transponder()
        ) * self.sector.weather(ship.loc).sensor_factor

    def compute_target_profile(self, target:core.SectorEntity, detector_or_distance_sq:Union[core.SectorEntity, float]) -> float:
        """ computes detector-specific profile of target """
        if isinstance(detector_or_distance_sq, float):
            distance_sq = detector_or_distance_sq
        else:
            distance_sq = target.phys.position.get_dist_sqrd(detector_or_distance_sq.phys.position)
        if distance_sq == 0.:
            distance_sq = 1.

        return self.compute_effective_profile(target) / (config.Settings.sensors.COEFF_DISTANCE * distance_sq)

    def detected(self, target:core.SectorEntity, detector:core.SectorEntity) -> bool:
        return self.compute_target_profile(target, detector) > self.compute_effective_threshold(detector)

    def spatial_query(self, detector:core.SectorEntity, bbox:Tuple[float, float, float, float]) -> Iterator[core.SectorEntity]:
        for hit in self.sector.spatial_query(bbox):
            if self.detected(hit, detector):
                yield hit

    def spatial_point(self, detector:core.SectorEntity, point:Union[Tuple[float, float], npt.NDArray[np.float64]], max_dist:Optional[float]=None) -> Iterator[core.SectorEntity]:
        for hit in self.sector.spatial_point(point, max_dist):
            if self.detected(hit, detector):
                yield hit

    def target(self, target:core.SectorEntity, detector:core.SectorEntity, notify_target:bool=True) -> core.AbstractSensorImage:
        if not self.detected(target, detector):
            raise ValueError(f'{detector} cannot detect {target}')
        image = SensorImage(target, detector, self)
        image.initialize()
        image.update(notify_target=notify_target)
        return image

    def sensor_ranges(self, entity:core.SectorEntity) -> Tuple[float, float, float]:
        # range to detect passive targets
        passive_profile = (config.Settings.sensors.COEFF_MASS * config.Settings.generate.SectorEntities.ship.MASS + config.Settings.sensors.COEFF_RADIUS * config.Settings.generate.SectorEntities.ship.RADIUS)
        # range to detect full thrust targets
        thrust_profile = config.Settings.sensors.COEFF_FORCE * config.Settings.generate.SectorEntities.ship.MAX_THRUST**config.Settings.sensors.FORCE_EXPONENT
        # range to detect active sesnsor targets
        sensor_profile = config.Settings.sensors.COEFF_SENSORS * config.Settings.generate.SectorEntities.ship.MAX_SENSOR_POWER

        threshold = self.compute_effective_threshold(entity)

        return (
            np.sqrt(passive_profile / threshold / config.Settings.sensors.COEFF_DISTANCE),
            np.sqrt(thrust_profile / threshold / config.Settings.sensors.COEFF_DISTANCE),
            np.sqrt(sensor_profile / threshold / config.Settings.sensors.COEFF_DISTANCE),
        )

    def profile_ranges(self, entity:core.SectorEntity) -> Tuple[float, float]:
        profile = self.compute_effective_profile(entity)

        # range we're detected by passive threats
        passive_threshold = config.Settings.sensors.COEFF_THRESHOLD / (config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)

        # range we're detected by active threats
        active_threshold = config.Settings.sensors.COEFF_THRESHOLD / (config.Settings.generate.SectorEntities.ship.MAX_SENSOR_POWER + config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)

        return (
            np.sqrt(profile / passive_threshold / config.Settings.sensors.COEFF_DISTANCE),
            np.sqrt(profile / active_threshold / config.Settings.sensors.COEFF_DISTANCE),
        )

    def compute_thrust_for_profile(self, ship:core.SectorEntity, distance_sq:float, threshold:float) -> float:
        # solve effective_profile equation for thrust (x below)

        # target_profile = effective_profile / (c * dist ** 2)
        # target_profile = y
        # y = (a + b + c_1 * x^p + d) * w / (c_2 * dist**2)
        # x = ((a + b + d) * w - y * (c_2 * dist**2) ) / (-c_1 * w) ** (1/e)
        profile_base = (
            config.Settings.sensors.COEFF_MASS * ship.mass +
            config.Settings.sensors.COEFF_RADIUS * ship.radius +
            config.Settings.sensors.COEFF_SENSORS * ship.sensor_settings.effective_sensor_power() +
            config.Settings.sensors.COEFF_TRANSPONDER * ship.sensor_settings.effective_transponder()
        ) * self.sector.weather(ship.loc).sensor_factor

        thrust_to_power = (threshold * config.Settings.sensors.COEFF_DISTANCE * distance_sq - profile_base) / (config.Settings.sensors.COEFF_FORCE)

        # TODO: this is a little janky because there might be no solution
        if thrust_to_power < 0.0:
            return -np.inf
        else:
            return pow(thrust_to_power, 1.0/config.Settings.sensors.FORCE_EXPONENT)

    def compute_thrust_for_sensor_power(self, ship:core.SectorEntity, distance_sq:float, sensor_power:float) -> float:
        threshold = config.Settings.sensors.COEFF_THRESHOLD / (sensor_power + config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)
        return self.compute_thrust_for_profile(ship, distance_sq, threshold)

class SensorImageManager:
    """ Manages SensorImage instances for a particular ship. """
    def __init__(self, ship:core.SectorEntity, sensor_image_ttl:float):
        self.ship = ship

        self._cached_entities:Dict[uuid.UUID, core.AbstractSensorImage] = {}
        self._cached_entities_by_idx:Dict[int, core.AbstractSensorImage]
        self._cached_entities_ts = -1.
        self._sensor_image_ttl = sensor_image_ttl
        self._sensor_loc_index = rtree.index.Index()

    @property
    def sensor_contacts(self) -> Mapping[uuid.UUID, core.AbstractSensorImage]:
        return self._cached_entities

    def spatial_point(self, point:Union[Tuple[float, float], npt.NDArray[np.float64]]) -> Iterable[core.AbstractSensorImage]:
        return (self._cached_entities_by_idx[x] for x in self._sensor_loc_index.nearest((point[0],point[1], point[0], point[1]), -1)) # type: ignore

    def spatial_query(self, bbox:Tuple[float, float, float, float]) -> Iterable[core.AbstractSensorImage]:
        return (self._cached_entities_by_idx[x] for x in self._sensor_loc_index.intersection(bbox)) # type: ignore

    def update(self) -> None:
        assert self.ship.sector

        # first we find all the detectable entities
        for hit in self.ship.sector.entities.values():
            if hit.entity_id in self._cached_entities:
                self._cached_entities[hit.entity_id].update(notify_target=False)
            elif self.ship.sector.sensor_manager.detected(hit, self.ship):
                self._cached_entities[hit.entity_id] = self.ship.sector.sensor_manager.target(hit, self.ship, notify_target=False)
        remove_ids:Set[uuid.UUID] = set()

        # then index all those images for retrieval later
        idx = 0
        self._sensor_loc_index = rtree.index.Index()
        self._cached_entities_by_idx = {}
        for image in self._cached_entities.values():
            if not image.is_active() or image.age > self._sensor_image_ttl:
                remove_ids.add(image.identity.entity_id)
            else:
                r = image.identity.radius
                loc = image.loc
                x,y= loc[0], loc[1]
                self._sensor_loc_index.insert(
                    idx,
                    (x-r, y-r,
                     x+r, y+r),
                )
                self._cached_entities_by_idx[idx] = image
                idx+=1
        for entity_id in remove_ids:
            del self._cached_entities[entity_id]
