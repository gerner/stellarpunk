""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

from typing import Tuple, Iterator, Optional
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
        ship.observe(self)
        self._target:Optional[core.SectorEntity] = target
        if target:
            target.observe(self)
            self._entity_id = target.entity_id
            self._short_id = target.short_id()
        else:
            self._entity_id = uuid.uuid4()
            self._short_id = f'UNK-{self._entity_id.hex[:8]}'
        self._last_update = 0.
        self._base_ts = 0.
        self._loc = ZERO_VECTOR
        self._velocity = ZERO_VECTOR
        self._is_active = True

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
    def entity_id(self) -> uuid.UUID:
        return self._entity_id

    def short_id(self) -> str:
        """ first 32 bits as hex """
        return self._short_id

    @property
    def age(self) -> float:
        return core.Gamestate.gamestate.timestamp - self._last_update

    @property
    def loc(self) -> npt.NDArray[np.float64]:
        return self._loc + self._velocity * (core.Gamestate.gamestate.timestamp - self._base_ts)

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        return self._velocity

    def is_active(self) -> bool:
        return self._is_active

    def update(self) -> bool:
        if self._target and self._ship:
            if self._sensor_manager.detected(self._target, self._ship):
                self._last_update = core.Gamestate.gamestate.timestamp
                self._base_ts = core.Gamestate.gamestate.timestamp
                self._loc = self._target.loc
                self._velocity = self._target.velocity
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
            self._entity_id = self._entity_id
            self._short_id = self._short_id
        return image

class SensorSettings(core.AbstractSensorSettings):
    def __init__(self, max_sensor_power:float=0.) -> None:
        self._max_sensor_power = max_sensor_power
        self._sensor_power = 0.
        self._transponder_on = False
        self._thrust = 0.

        # keeps track of history on sensor spikes
        self._last_sensor_power = 0.
        self._last_sensor_power_ts = 0.
        self._last_transponder_ts = 0.
        self._last_thrust = 0.
        self._last_thrust_ts = 0.

    @property
    def sensor_power(self) -> float:
        return self._sensor_power
    @property
    def transponder(self) -> bool:
        return self._transponder_on

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
    def effective_threshold(self) -> float:
        """ computes the sensor threshold accounting for sensor power """
        return config.Settings.sensors.COEFF_THRESHOLD / (self.effective_sensor_power() + config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)

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
        self._last_thrust_ts = core.Gamestate.gamestate.timestamp
        self._thrust = thrust

class SensorManager(core.AbstractSensorManager):
    """ Models sensors for a sector """

    def __init__(self, sector:core.Sector):
        self.sector = sector

    def compute_effective_profile(self, ship:core.SectorEntity) -> float:
        """ computes the profile  accounting for ship and sector factors """
        return (
            config.Settings.sensors.COEFF_MASS * ship.mass +
            config.Settings.sensors.COEFF_RADIUS * ship.radius +
            config.Settings.sensors.COEFF_FORCE * ship.sensor_settings.effective_thrust() +
            config.Settings.sensors.COEFF_SENSORS * ship.sensor_settings.effective_sensor_power() +
            config.Settings.sensors.COEFF_TRANSPONDER * ship.sensor_settings.effective_transponder()
        ) * self.sector.weather_factor

    def compute_target_profile(self, target:core.SectorEntity, detector:core.SectorEntity) -> float:
        """ computes detector-specific profile of target """
        distance_sq = target.phys.position.get_dist_sqrd(detector.phys.position)
        if distance_sq == 0.:
            distance_sq = 1.

        return self.compute_effective_profile(target) / (config.Settings.sensors.COEFF_DISTANCE * distance_sq)

    def detected(self, target:core.SectorEntity, detector:core.SectorEntity) -> bool:
        return self.compute_target_profile(target, detector) > detector.sensor_settings.effective_threshold()

    def spatial_query(self, detector:core.SectorEntity, bbox:Tuple[float, float, float, float]) -> Iterator[core.SectorEntity]:
        for hit in self.sector.spatial_query(bbox):
            if self.detected(hit, detector):
                yield hit

    def target(self, target:core.SectorEntity, detector:core.SectorEntity) -> core.AbstractSensorImage:
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

        threshold = entity.sensor_settings.effective_threshold()

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
