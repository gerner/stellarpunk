""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

from typing import Tuple, Iterator

import numpy.typing as npt
import numpy as np

from stellarpunk import core, config, util

class SensorImage(core.AbstractSensorImage):
    def __init__(self, target:core.SectorEntity) -> None:
        self._target = target
        self._last_update = core.Gamestate.gamestate.timestamp
        self._loc = target.loc
        self._velocity = target.velocity

    @property
    def age(self) -> float:
        return core.Gamestate.gamestate.timestamp - self._last_update

    @property
    def loc(self) -> npt.NDArray[np.float64]:
        return self._loc + self._velocity * (core.Gamestate.gamestate.timestamp - self._last_update)

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        return self._velocity

    def update(self) -> None:
        self._last_update = core.Gamestate.gamestate.timestamp
        self._loc = self._target.loc
        self._velocity = self._target.velocity

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
