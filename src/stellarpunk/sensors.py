""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

from typing import Tuple, Iterator, Optional

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
                return True
            else:
                return False
        else:
            return False

class SensorManager(core.AbstractSensorManager):
    """ Models sensors for a sector """

    def __init__(self, sector:core.Sector):
        self.sector = sector

    def compute_decayed_sensor_power(self, entity:core.SectorEntity) -> float:
        if entity.last_sensor_power == entity.sensor_power:
            return entity.sensor_power
        return (entity.last_sensor_power - entity.sensor_power) * config.Settings.sensors.DECAY_SENSORS ** (core.Gamestate.gamestate.timestamp - entity.last_sensor_power_ts) + entity.sensor_power

    def compute_decayed_transponder(self, entity:core.SectorEntity) -> float:
        if entity.transponder_on:
            return 1.0
        return config.Settings.sensors.DECAY_TRANSPONDER ** (core.Gamestate.gamestate.timestamp - entity.last_transponder_ts)

    def compute_effective_profile(self, ship:core.SectorEntity) -> float:
        """ computes the profile  accounting for ship and sector factors """
        return (config.Settings.sensors.COEFF_MASS * ship.mass + config.Settings.sensors.COEFF_RADIUS * ship.radius + config.Settings.sensors.COEFF_FORCE * ship.phys.force.length + config.Settings.sensors.COEFF_SENSORS * self.compute_decayed_sensor_power(ship) + config.Settings.sensors.COEFF_TRANSPONDER * self.compute_decayed_transponder(ship)) * self.sector.weather_factor

    def compute_target_profile(self, target:core.SectorEntity, detector:core.SectorEntity) -> float:
        """ computes detector-specific profile of target """
        distance = util.distance(target.loc, detector.loc)
        if distance == 0.:
            distance = 1.

        return self.compute_effective_profile(target) / (config.Settings.sensors.COEFF_DISTANCE * distance ** 2.)

    def compute_sensor_threshold(self, ship:core.SectorEntity) -> float:
        """ computes the sensor threshold accounting for sensor power """
        return config.Settings.sensors.COEFF_THRESHOLD / (self.compute_decayed_sensor_power(ship) + config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)

    def detected(self, target:core.SectorEntity, detector:core.SectorEntity) -> bool:
        return self.compute_target_profile(target, detector) > self.compute_sensor_threshold(detector)

    def spatial_query(self, detector:core.SectorEntity, bbox:Tuple[float, float, float, float]) -> Iterator[core.SectorEntity]:
        for hit in self.sector.spatial_query(bbox):
            if self.detected(hit, detector):
                yield hit

    def target(self, target:core.SectorEntity, detector:core.SectorEntity) -> core.AbstractSensorImage:
        image = SensorImage(target, detector, self)
        image.update()
        return image

    def set_sensors(self, entity:core.SectorEntity, level:float) -> None:
        # keep track of spikes in sensor usage so the impact on profile can
        # decay with time
        new_level = level * entity.max_sensor_power
        if new_level == entity.sensor_power:
            return
        decayed_power = self.compute_decayed_sensor_power(entity)
        entity.last_sensor_power = decayed_power
        entity.last_sensor_power_ts = core.Gamestate.gamestate.timestamp
        entity.sensor_power = new_level

    def set_transponder(self, entity:core.SectorEntity, on:bool) -> None:
        # keep track of the last time the transponder was on so it's impact to
        # profile can decay with time
        if entity.transponder_on:
            entity.last_transponder_ts = core.Gamestate.gamestate.timestamp
        entity.transponder_on = on

    def sensor_ranges(self, entity:core.SectorEntity) -> Tuple[float, float, float]:
        # passive
        passive_profile = (config.Settings.sensors.COEFF_MASS * config.Settings.generate.SectorEntities.ship.MASS + config.Settings.sensors.COEFF_RADIUS * config.Settings.generate.SectorEntities.ship.RADIUS) * self.sector.weather_factor
        # full thrust
        thrust_profile = config.Settings.sensors.COEFF_FORCE * config.Settings.generate.SectorEntities.ship.MAX_THRUST * self.sector.weather_factor
        # max sesnsors
        sensor_profile = config.Settings.sensors.COEFF_SENSORS * config.Settings.generate.SectorEntities.ship.MAX_SENSOR_POWER * self.sector.weather_factor

        threshold = self.compute_sensor_threshold(entity)

        return (
            np.sqrt(passive_profile / threshold / config.Settings.sensors.COEFF_DISTANCE),
            np.sqrt(thrust_profile / threshold / config.Settings.sensors.COEFF_DISTANCE),
            np.sqrt(sensor_profile / threshold / config.Settings.sensors.COEFF_DISTANCE),
        )
