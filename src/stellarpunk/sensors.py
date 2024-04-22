""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

from typing import Tuple, Iterator, Optional, Any, Union
import uuid

import numpy.typing as npt
import numpy as np

from stellarpunk import core, config, util

ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False

class SensorImage(core.AbstractSensorImage, core.SectorEntityObserver):
    def __init__(self, target:Optional[core.SectorEntity], ship:core.SectorEntity, sensor_manager:"SensorManager") -> None:
        self._sensor_manager = sensor_manager
        self._ship:Optional[core.SectorEntity] = ship
        self._detector_entity_id = ship.entity_id
        self._detector_short_id = ship.short_id()
        self._target:Optional[core.SectorEntity] = target
        if target:
            self._target_entity_id = target.entity_id
            self._target_short_id = target.short_id()
            target.observe(self)
        else:
            # we should never need these. the only valid case where we don't
            # have a target is if we're copied from another SensorImage in
            # which case we should have gotten their _target_entity_id /
            # _short_id
            self._target_entity_id = uuid.uuid4()
            self._target_short_id = f'UNK-{self._target_entity_id.hex[:8]}'
        ship.observe(self)
        self._last_update = 0.
        self._last_profile = 0.
        self._loc = ZERO_VECTOR
        self._velocity = ZERO_VECTOR
        self._is_active = True

        self._loc_bias = np.array((0.0, 0.0))
        self._velocity_bias = np.array((0.0, 0.0))
        self._last_bias_update_ts = 0

    def __str__(self) -> str:
        return f'{self._target_short_id} detected by {self._detector_short_id} {self.age}s old'

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, core.AbstractSensorImage):
            return False
        return self._target_entity_id == other.target_entity_id and self._detector_entity_id == other.detector_entity_id

    def __hash__(self) -> int:
        return hash((self._target_entity_id, self._detector_entity_id))

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self._target:
            # might be risky checking if detected on logically destroyed entity
            if self._ship and self._sensor_manager.detected(entity, self._ship):
                self._is_active = False
            self._target = None
            if self._ship:
                self._ship.unobserve(self)
            self._ship = None
        elif entity == self._ship:
            if self._target:
                self._target.unobserve(self)
                self._target = None
            self._ship = None
        else:
            raise ValueError(f'got entity_destroyed for unexpected entity {entity}')

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if entity == self._target:
            # might be risky checking if detected on logically migrated entity
            if self._ship and self._sensor_manager.detected(entity, self._ship):
                self._is_active = False
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
    def target_entity_id(self) -> uuid.UUID:
        return self._target_entity_id
    def target_short_id(self) -> str:
        return self._target_short_id
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
        return self._loc + self._velocity * (core.Gamestate.gamestate.timestamp - self._last_update)

    @property
    def profile(self) -> float:
        return self._last_profile

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        return self._velocity

    def is_active(self) -> bool:
        return self._is_active

    def update(self) -> bool:
        if self._target and self._ship:
            if self._sensor_manager.detected(self._target, self._ship):
                self._last_profile = self._sensor_manager.compute_target_profile(self._target, self._ship)
                self._last_update = core.Gamestate.gamestate.timestamp
                self._loc = self._target.loc
                self._velocity = np.array(self._target.velocity)

                # let the target know they've been targeted by us
                if self._sensor_manager.detected(self._ship, self._target):
                    self._target.target(self._ship)
                return True
            else:
                return False
        else:
            return False

    def copy(self, detector:core.SectorEntity) -> core.AbstractSensorImage:
        image = SensorImage(self._target, detector, self._sensor_manager)
        if self._target is None:
            self._target_entity_id = self._target_entity_id
            self._target_short_id = self._target_short_id
        image._last_update = self._last_update
        image._last_profile = self._last_profile
        image._loc = self._loc
        image._velocity = self._velocity
        image._is_active = self._is_active
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


class SensorManager(core.AbstractSensorManager):
    """ Models sensors for a sector """

    def __init__(self, sector:core.Sector):
        self.sector = sector

    def compute_effective_threshold(self, ship:core.SectorEntity) -> float:
        return ship.sensor_settings.effective_threshold() * self.sector.weather_factor

    def compute_effective_profile(self, ship:core.SectorEntity) -> float:
        """ computes the profile  accounting for ship and sector factors """
        return (
            config.Settings.sensors.COEFF_MASS * ship.mass +
            config.Settings.sensors.COEFF_RADIUS * ship.radius +
            config.Settings.sensors.COEFF_FORCE * ship.sensor_settings.effective_thrust() +
            config.Settings.sensors.COEFF_SENSORS * ship.sensor_settings.effective_sensor_power() +
            config.Settings.sensors.COEFF_TRANSPONDER * ship.sensor_settings.effective_transponder()
        ) * self.sector.weather_factor

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

    def target(self, target:core.SectorEntity, detector:core.SectorEntity) -> core.AbstractSensorImage:
        if not self.detected(target, detector):
            raise ValueError(f'{detector} cannot detect {target}')
        image = SensorImage(target, detector, self)
        image.update()
        return image

    def sensor_ranges(self, entity:core.SectorEntity) -> Tuple[float, float, float]:
        # range to detect passive targets
        passive_profile = (config.Settings.sensors.COEFF_MASS * config.Settings.generate.SectorEntities.ship.MASS + config.Settings.sensors.COEFF_RADIUS * config.Settings.generate.SectorEntities.ship.RADIUS) * self.sector.weather_factor
        # range to detect full thrust targets
        thrust_profile = config.Settings.sensors.COEFF_FORCE * config.Settings.generate.SectorEntities.ship.MAX_THRUST * self.sector.weather_factor
        # range to detect active sesnsor targets
        sensor_profile = config.Settings.sensors.COEFF_SENSORS * config.Settings.generate.SectorEntities.ship.MAX_SENSOR_POWER * self.sector.weather_factor

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
        # solve effective_profile equation for thrust

        # target_profile = effective_profile / (c * dist ** 2)
        # y = (a + b + c_1 * x + d) * w / (c_2 * dist**2)
        # x = ((a + b + d) * w - y * (c_2 * dist**2) ) / (-c_1 * w)
        profile_base = (
            config.Settings.sensors.COEFF_MASS * ship.mass +
            config.Settings.sensors.COEFF_RADIUS * ship.radius +
            config.Settings.sensors.COEFF_SENSORS * ship.sensor_settings.effective_sensor_power() +
            config.Settings.sensors.COEFF_TRANSPONDER * ship.sensor_settings.effective_transponder()
        ) * self.sector.weather_factor

        return (threshold * config.Settings.sensors.COEFF_DISTANCE * distance_sq - profile_base) / (config.Settings.sensors.COEFF_FORCE * self.sector.weather_factor)

    def compute_thrust_for_sensor_power(self, ship:core.SectorEntity, distance_sq:float, sensor_power:float) -> float:
        threshold = config.Settings.sensors.COEFF_THRESHOLD / (sensor_power + config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)
        return self.compute_thrust_for_profile(ship, distance_sq, threshold)
