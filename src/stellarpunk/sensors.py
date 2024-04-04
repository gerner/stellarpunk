""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

from typing import Tuple, Iterator, Optional

import numpy.typing as npt
import numpy as np

from stellarpunk import core, config, util

class SensorImage(core.AbstractSensorImage, core.SectorEntityObserver):
    def __init__(self, target:core.SectorEntity, ship:core.SectorEntity, sensor_manager:"SensorManager") -> None:
        self._sensor_manager = sensor_manager
        self._ship:Optional[core.SectorEntity] = ship
        ship.observe(self)
        self._target:Optional[core.SectorEntity] = target
        target.observe(self)
        self._last_update = core.Gamestate.gamestate.timestamp
        self._loc = target.loc
        self._velocity = target.velocity
        self._is_active = True

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        assert self._target
        assert self._ship
        if entity == self._target:
            if self._sensor_manager.detected(self._target, self._ship):
                self._is_active = False
            self._target = None
            self._ship.unobserve(self)
            self._ship = None
        elif entity == self._ship:
            self._target.unobserve(self)
            self._target = None
            self._ship = None
        else:
            raise ValueError(f'got entity_destroyed for unexpected entity {entity}')

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        assert self._target
        assert self._ship
        if entity == self._target:
            if self._sensor_manager.detected(self._target, self._ship):
                self._is_active = False
            self._target.unobserve(self)
            self._target = None
            self._ship.unobserve(self)
            self._ship = None
        elif entity == self._ship:
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
        return self._loc + self._velocity * (core.Gamestate.gamestate.timestamp - self._last_update)

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        return self._velocity

    def is_active(self) -> bool:
        return self._is_active

    def update(self) -> bool:
        if self._target and self._ship and self._sensor_manager.detected(self._target, self._ship):
            self._last_update = core.Gamestate.gamestate.timestamp
            self._loc = self._target.loc
            self._velocity = self._target.velocity
            return True
        else:
            assert self._target is None
            assert self._ship is None
            return False

class SensorManager(core.AbstractSensorManager):
    """ Models sensors for a sector """

    def __init__(self, sector:core.Sector):
        self.sector = sector

    def compute_effective_profile(self, ship:core.SectorEntity) -> float:
        """ computes the profile  accounting for ship and sector factors """
        return (config.Settings.sensors.COEFF_MASS * ship.mass + config.Settings.sensors.COEFF_RADIUS * ship.radius + config.Settings.sensors.COEFF_FORCE * ship.phys.force.length + config.Settings.sensors.COEFF_SENSORS * ship.sensor_power + config.Settings.sensors.COEFF_TRANSPONDER * ship.transponder_on) * self.sector.weather_factor

    def compute_target_profile(self, target:core.SectorEntity, detector:core.SectorEntity) -> float:
        """ computes detector-specific profile of target """
        distance = util.distance(target.loc, detector.loc)
        if distance == 0.:
            distance = 1.

        return self.compute_effective_profile(target) / (config.Settings.sensors.COEFF_DISTANCE * distance ** 2.)

    def compute_sensor_threshold(self, ship:core.SectorEntity) -> float:
        """ computes the sensor threshold accounting for sensor power """
        return config.Settings.sensors.COEFF_THRESHOLD / (ship.sensor_power + config.Settings.sensors.COEFF_THRESHOLD/config.Settings.sensors.INTERCEPT_THRESHOLD)

    def detected(self, target:core.SectorEntity, detector:core.SectorEntity) -> bool:
        return self.compute_target_profile(target, detector) > self.compute_sensor_threshold(detector)

    def spatial_query(self, detector:core.SectorEntity, bbox:Tuple[float, float, float, float]) -> Iterator[core.SectorEntity]:
        for hit in self.sector.spatial_query(bbox):
            if self.detected(hit, detector):
                yield hit

    def target(self, target:core.SectorEntity, detector:core.SectorEntity) -> core.AbstractSensorImage:
        return SensorImage(target, detector, self)
