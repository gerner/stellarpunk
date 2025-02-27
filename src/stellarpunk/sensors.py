""" Sensor handling stuff, limiting what's visible and adding in ghosts. """

import uuid
import logging
import weakref
import enum
import collections
import itertools
from collections.abc import Collection, Mapping, MutableMapping, Iterator, Iterable
from typing import Optional, Any, Union

import rtree.index
import numpy.typing as npt
import numpy as np

from stellarpunk import core, config, util, events
from stellarpunk.core import sector_entity

logger = logging.getLogger(__name__)

ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False

class Events(enum.IntEnum):
    SCANNED = enum.auto()
    TARGETED = enum.auto()
    IDENTIFIED = enum.auto()
    INACTIVE = enum.auto()

class ContextKeys(enum.IntEnum):
    DETECTOR = enum.auto()
    TARGET = enum.auto()
    SECTOR = enum.auto()
    STATIC_COUNT = enum.auto()
    DYNAMIC_COUNT = enum.auto()
    INACTIVE_REASON = enum.auto()

class SensorImage(core.AbstractSensorImage):
    @classmethod
    def create_sensor_image(cls, target:Optional[core.SectorEntity], ship:core.SectorEntity, sensor_manager:core.AbstractSensorManager, identity:Optional[core.SensorIdentity]=None, identified:bool=False) -> "SensorImage":
        if identity is None:
            assert(target)
            identity = core.SensorIdentity(target)
        image = SensorImage(identity)
        image.set_sensor_manager(sensor_manager)
        image._ship = ship
        image._detector_id = ship.entity_id
        image._identified = identified
        return image

    def __init__(self, identity:core.SensorIdentity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self._identity = identity
        self._sensor_manager:core.AbstractSensorManager = None # type: ignore
        self._ship:core.SectorEntity = None # type: ignore
        self._detector_id:uuid.UUID = None # type: ignore
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
        return f'{self._identity.short_id()} tracked by {self._ship or self._detector_id} {self.age}s old'

    def __hash__(self) -> int:
        return hash((self._identity.entity_id, self._detector_id))

    def _detection_range(self) -> float:
        """ Range at which we should be able to detect this target for sure

        This is a lower bound, target may be detectable at a longer range."""
        passive_profile = (config.Settings.sensors.COEFF_MASS * self._identity.mass + config.Settings.sensors.COEFF_RADIUS * self._identity.radius)
        threshold = self._sensor_manager.compute_effective_threshold(self._ship)
        return np.sqrt(passive_profile / threshold / config.Settings.sensors.COEFF_DISTANCE)

    def target_destroyed(self, entity:core.SectorEntity) -> None:
        assert entity.entity_id == self._identity.entity_id
        # might be risky checking if detected on logically destroyed entity
        if self._ship and self._sensor_manager.detected(entity, self._ship):
            self._is_active = False
            self._inactive_reason = core.SensorImageInactiveReason.DESTROYED
            crew = core.crew(self._ship)
            gamestate = core.Gamestate.gamestate
            gamestate.trigger_event(
                    crew,
                    gamestate.event_manager.e(Events.INACTIVE),
                    {
                        gamestate.event_manager.ck(ContextKeys.DETECTOR): self._ship.short_id_int(),
                        gamestate.event_manager.ck(ContextKeys.TARGET): self._identity.short_id_int(),
                        gamestate.event_manager.ck(ContextKeys.INACTIVE_REASON): self._inactive_reason,
                    }
            )

    def target_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        assert entity.entity_id == self._identity.entity_id
        # might be risky checking if detected on logically migrated entity
        if self._ship and self._sensor_manager.detected(entity, self._ship):
            self._is_active = False
            self._inactive_reason = core.SensorImageInactiveReason.MIGRATED
            crew = core.crew(self._ship)
            gamestate = core.Gamestate.gamestate
            gamestate.trigger_event(
                    crew,
                    gamestate.event_manager.e(Events.INACTIVE),
                    {
                        gamestate.event_manager.ck(ContextKeys.DETECTOR): self._ship.short_id_int(),
                        gamestate.event_manager.ck(ContextKeys.TARGET): self._identity.short_id_int(),
                        gamestate.event_manager.ck(ContextKeys.INACTIVE_REASON): self._inactive_reason,
                    }
            )


    def set_sensor_manager(self, sensor_manager:core.AbstractSensorManager) -> None:
        self._sensor_manager = sensor_manager

    @property
    def age(self) -> float:
        return core.Gamestate.gamestate.timestamp - self._last_update

    @property
    def loc(self) -> npt.NDArray[np.float64]:
        if self._ship.sensor_settings.ignore_bias:
            return (self._loc) + (self._velocity) * (core.Gamestate.gamestate.timestamp - self._last_update)
        else:
            return (self._loc + self._loc_bias) + (self._velocity + self._velocity_bias) * (core.Gamestate.gamestate.timestamp - self._last_update)

    @property
    def profile(self) -> float:
        return self._last_profile

    @property
    def fidelity(self) -> float:
        return self.profile / self._sensor_manager.compute_effective_threshold(self._ship)

    @property
    def detected(self) -> bool:
        if not self._ship.sector:
            return False
        if self.identity.entity_id not in self._ship.sector.entities:
            return False
        target = self._ship.sector.entities[self.identity.entity_id]
        return self._sensor_manager.detected(target, self._ship)

    @property
    def currently_identified(self) -> bool:
        return self.fidelity * config.Settings.sensors.COEFF_IDENTIFICATION_FIDELITY > 1.0

    @property
    def identified(self) -> bool:
        return self._identified

    @property
    def transponder(self) -> bool:
        if not self._ship.sector:
            return False
        if self.identity.entity_id not in self._ship.sector.entities:
            return False
        target = self._ship.sector.entities[self.identity.entity_id]
        return target.sensor_settings.transponder

    @property
    def identity(self) -> core.SensorIdentity:
        return self._identity

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        if self._ship.sensor_settings.ignore_bias:
            return self._velocity
        else:
            return self._velocity + self._velocity_bias

    @property
    def acceleration(self) -> npt.NDArray[np.float64]:
        return self._acceleration

    def is_active(self) -> bool:
        return self._is_active

    @property
    def inactive_reason(self) -> core.SensorImageInactiveReason:
        return self._inactive_reason

    def initialize(self, target:core.SectorEntity) -> None:
        self._initialize_bias(target)

    def destroy(self) -> None:
        self._ship = None # type: ignore

    def _initialize_bias(self, target:core.SectorEntity) -> None:
        # choose a direction for the location bias
        # choose an angle and radius offset relative to detector
        distance = util.magnitude(*(target.loc - self._ship.loc))
        if distance > 0.0:
            offset_r = core.Gamestate.gamestate.random.uniform(low=-1.0, high=1.0) * config.Settings.sensors.COEFF_BIAS_OFFSET_R * distance
            offset_theta = core.Gamestate.gamestate.random.uniform(low=-1.0, high=1.0) * config.Settings.sensors.COEFF_BIAS_OFFSET_THETA

            target_r, target_theta = util.cartesian_to_polar(*(target.loc - self._ship.loc))

            self._loc_bias_direction = np.array(util.polar_to_cartesian(target_r+offset_r, target_theta+offset_theta)) + self._ship.loc - target.loc
            self._loc_bias_direction /= util.magnitude(*self._loc_bias_direction)
        self._last_bias_update_ts = -np.inf

    def _update_bias(self, target:core.SectorEntity) -> None:
        if isinstance(target, sector_entity.Projectile):
            self._last_bias_update_ts = core.Gamestate.gamestate.timestamp
            return

        new_loc_bias = self._loc_bias_direction / self.fidelity * config.Settings.sensors.COEFF_BIAS_LOC

        #TODO: add some extra noise to new_loc_bias?

        time_delta = core.Gamestate.gamestate.timestamp - self._last_bias_update_ts
        if self.fidelity < self._prior_fidelity:
            alpha = 1/(-(1/config.Settings.sensors.COEFF_BIAS_TIME_DECAY_DOWN * time_delta + 1)) + 1
        else:
            alpha = 1/(-(1/config.Settings.sensors.COEFF_BIAS_TIME_DECAY_UP * time_delta + 1)) + 1

        new_loc_bias = alpha * new_loc_bias + (1-alpha) * self._loc_bias
        #new_velocity_bias = np.array((0.0, 0.0))

        self._loc_bias = new_loc_bias
        #self._velocity_bias = new_velocity_bias
        self._last_bias_update_ts = core.Gamestate.gamestate.timestamp

    def update(self, notify_target:bool=True) -> bool:
        target:Optional[core.SectorEntity] = None
        if self._ship.sector:
            # try to re-fetch the target if possible
            # this might happen if this image is created from intel
            # also possible if the target jumps away without us detecting them
            # and then comes back
            if self.identity.entity_id in self._ship.sector.entities:
                target = self._ship.sector.entities[self.identity.entity_id]

        if target:
            # update sensor reading if possible
            if self._sensor_manager.detected(target, self._ship):
                self._prior_fidelity = self.fidelity
                self._identity.angle = target.angle
                self._last_profile = self._sensor_manager.compute_target_profile(target, self._ship)
                self._update_bias(target)

                since_last_update = core.Gamestate.gamestate.timestamp - self._last_update
                if since_last_update < config.Settings.sensors.ACCEL_PREDICTION_MAX_SEC:
                    self._acceleration = target.velocity - self._velocity
                else:
                    self._acceleration = ZERO_VECTOR

                self._last_update = core.Gamestate.gamestate.timestamp
                self._loc = target.loc
                self._velocity = np.array(target.velocity)

                if not self._identified and self.fidelity * config.Settings.sensors.COEFF_IDENTIFICATION_FIDELITY > 1.0:
                    crew = core.crew(self._ship)
                    if crew:
                        #TODO: what about passengers? should they get this event too?
                        gamestate = core.Gamestate.gamestate
                        gamestate.trigger_event(
                                crew,
                                gamestate.event_manager.e(Events.IDENTIFIED),
                                {
                                    gamestate.event_manager.ck(ContextKeys.DETECTOR): self._ship.short_id_int(),
                                    gamestate.event_manager.ck(ContextKeys.TARGET): target.short_id_int(),
                                }
                        )
                    # once identified, always identified
                    self._identified = True

                # let the target know they've been targeted by us
                if notify_target and self._sensor_manager.detected(self._ship, target):
                    #TODO: this is going to get triggered a lot. should we only
                    # do it the first time they get targeted?
                    #TODO: what about passengers? should they get this event too?
                    candidates = list(itertools.chain(core.crew(self._ship), core.crew(target)))
                    if candidates:
                        crew = core.crew(self._ship)
                        gamestate = core.Gamestate.gamestate
                        gamestate.trigger_event(
                                crew,
                                gamestate.event_manager.e(Events.INACTIVE),
                                {
                                    gamestate.event_manager.ck(ContextKeys.DETECTOR): self._ship.short_id_int(),
                                    gamestate.event_manager.ck(ContextKeys.TARGET): self._identity.short_id_int(),
                                    gamestate.event_manager.ck(ContextKeys.INACTIVE_REASON): self._inactive_reason,
                                }
                        )

                    target.target(self._ship)
                return True
            else:
                self._acceleration = ZERO_VECTOR
                return False
        else:
            if self._is_active:
                # we did not observe the target becoming inactive
                # check to see if we have enough information to deduce it is
                # inactive now, i.e. we're close enough
                if util.distance(self._ship.loc, self.loc) < self._detection_range():
                    self._is_active = False
                    self._inactive_reason = core.SensorImageInactiveReason.MISSING
                    crew = core.crew(self._ship)
                    gamestate = core.Gamestate.gamestate
                    gamestate.trigger_event(
                            crew,
                            gamestate.event_manager.e(Events.INACTIVE),
                            {
                                gamestate.event_manager.ck(ContextKeys.DETECTOR): self._ship.short_id_int(),
                                gamestate.event_manager.ck(ContextKeys.TARGET): self._identity.short_id_int(),
                            }
                    )


            return False

    def copy(self, detector:core.SectorEntity) -> core.AbstractSensorImage:
        """ Retargets the underlying craft with a (potentially) new detector

        This does not reset state, but future updates will be with respect to
        the new detector. """
        if detector.sensor_settings.has_image(self.identity.entity_id):
            return detector.sensor_settings.get_image(self.identity.entity_id)

        identity = self._identity
        image = SensorImage.create_sensor_image(None, detector, self._sensor_manager, identity=self._identity, identified=self._identified)
        image._last_update = self._last_update
        image._last_profile = self._last_profile
        image._loc = self._loc
        image._velocity = self._velocity
        image._acceleration = self._acceleration
        image._is_active = self._is_active

        image._loc_bias = self._loc_bias
        image._velocity_bias = self._velocity_bias
        image._last_bias_update_ts = self._last_bias_update_ts
        detector.sensor_settings.register_image(image)

        assert image._ship

        return image

class SensorSettings(core.SectorEntityObserver, core.AbstractSensorSettings):
    def __init__(self, max_sensor_power:float=0., sensor_intercept:float=100.0, initial_sensor_power:Optional[float]=None, initial_transponder:bool=True) -> None:
        super().__init__()
        if initial_sensor_power is None:
            initial_sensor_power = max_sensor_power
        if initial_sensor_power > max_sensor_power or 0.0 > initial_sensor_power:
            raise ValueError(f'0 must be <= {initial_sensor_power=} <= {max_sensor_power=}')

        self._detector_id:uuid.UUID = None # type: ignore

        self._max_sensor_power = max_sensor_power
        self._sensor_intercept = sensor_intercept
        self._sensor_power = initial_sensor_power
        self._transponder_on = initial_transponder
        self._thrust = 0.

        # keeps track of history on sensor spikes
        self._last_sensor_power = self._sensor_power
        self._last_sensor_power_ts = 0.
        self._last_transponder_ts = -np.inf
        self._last_thrust = 0.
        self._last_thrust_ts = 0.

        self._thrust_seconds = 0.

        self._ignore_bias = False

        self._images:weakref.WeakValueDictionary[uuid.UUID, core.AbstractSensorImage] = weakref.WeakValueDictionary()

        self._cached_effective_profile:float = 0.0
        self._cached_effective_profile_ts:float = -np.inf

    # core.SectorEntityObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return self._detector_id

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity.entity_id in self._images:
            self._images[entity.entity_id].target_destroyed(entity)
        entity.unobserve(self)

    def entity_pre_migrate(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if entity.entity_id in self._images:
            self._images[entity.entity_id].target_migrated(entity, from_sector, to_sector)
        entity.unobserve(self)

    def set_detector_id(self, detector_id:uuid.UUID) -> None:
        self._detector_id = detector_id

    def register_image(self, image:core.AbstractSensorImage) -> None:
        if core.Gamestate.gamestate.contains_entity(image.identity.entity_id):
            target = core.Gamestate.gamestate.get_entity(image.identity.entity_id, core.SectorEntity)
            target.observe(self)
        self._images[image.identity.entity_id] = image
    def unregister_image(self, image:core.AbstractSensorImage) -> None:
        if core.Gamestate.gamestate.contains_entity(image.identity.entity_id):
            target = core.Gamestate.gamestate.get_entity(image.identity.entity_id, core.SectorEntity)
            target.unobserve(self)
        del self._images[image.identity.entity_id]
    def has_image(self, target_id:uuid.UUID) -> bool:
        return target_id in self._images
    def get_image(self, target_id:uuid.UUID) -> core.AbstractSensorImage:
        return self._images[target_id]

    @property
    def images(self) -> Collection[core.AbstractSensorImage]:
        return list(self._images.values())
    def clear_images(self) -> None:
        for image in self._images.values():
            image.destroy()
        self._images.clear()
        self.clear_observings()

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

    def effective_profile(self, sector:core.Sector, ship:core.SectorEntity) -> float:
        if core.Gamestate.gamestate.timestamp > self._cached_effective_profile_ts + config.Settings.sensors.EFFECTIVE_PROFILE_CACHE_TTL:
            self._cached_effective_profile_ts = core.Gamestate.gamestate.timestamp
            self._cached_effective_profile = (
                config.Settings.sensors.COEFF_MASS * ship.mass +
                config.Settings.sensors.COEFF_RADIUS * ship.radius +
                config.Settings.sensors.COEFF_FORCE * self.effective_thrust()**config.Settings.sensors.FORCE_EXPONENT +
                config.Settings.sensors.COEFF_SENSORS * self.effective_sensor_power() +
                config.Settings.sensors.COEFF_TRANSPONDER * self.effective_transponder()
            ) * sector.weather(ship.loc).sensor_factor
        return self._cached_effective_profile

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
        return ship.sensor_settings.effective_profile(self.sector, ship)

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

    def scan(self, detector:core.SectorEntity) -> Iterable[core.AbstractSensorImage]:
        """ Scans the sector for detectable sensor images

        This does not notify targets of being targeted. Instead any targets of
        interest should be retargeted after the scan. Think of this as a
        passive scan looking for shallow information about surroundings.

        This is an expensive operation. It has to touch every sector entity in
        the sector. So don't do this too often.  """

        static_hits = 0
        dynamic_hits = 0
        hits:list[core.AbstractSensorImage] = []
        for hit in self.sector.entities.values():
            if self.detected(hit, detector):
                target = self.target(hit, detector, notify_target=False)
                hits.append(target)
                if target.identified:
                    target_entity = core.Gamestate.gamestate.get_entity(target.identity.entity_id, target.identity.object_type)
                    if target_entity.is_static:
                        static_hits += 1
                    else:
                        dynamic_hits += 1

        # trigger an event that we've done the scan
        crew = core.crew(detector)
        if crew:
            gamestate = core.Gamestate.gamestate
            gamestate.trigger_event(
                    crew,
                    gamestate.event_manager.e(Events.SCANNED),
                    {
                        gamestate.event_manager.ck(ContextKeys.DETECTOR): detector.short_id_int(),
                        gamestate.event_manager.ck(ContextKeys.SECTOR): self.sector.short_id_int(),
                        gamestate.event_manager.ck(ContextKeys.STATIC_COUNT): static_hits,
                        gamestate.event_manager.ck(ContextKeys.DYNAMIC_COUNT): dynamic_hits,
                    },
                    merge_key=detector.entity_id,
            )

        return hits

    def target(self, target:core.SectorEntity, detector:core.SectorEntity, notify_target:bool=True) -> core.AbstractSensorImage:
        if not self.detected(target, detector):
            raise ValueError(f'{detector} cannot detect {target}')
        if detector.sensor_settings.has_image(target.entity_id):
            image = detector.sensor_settings.get_image(target.entity_id)
            image.update(notify_target=notify_target)
            return image
        else:
            image = SensorImage.create_sensor_image(target, detector, self)
            image.initialize(target)
            image.update(notify_target=notify_target)
            detector.sensor_settings.register_image(image)
            return image

    def target_from_identity(self, target_identity:core.SensorIdentity, detector:core.SectorEntity, loc:npt.NDArray[np.float64], notify_target:bool=True, identified:bool=True) -> core.AbstractSensorImage:
        if detector.sensor_settings.has_image(target_identity.entity_id):
            return detector.sensor_settings.get_image(target_identity.entity_id)

        image = SensorImage.create_sensor_image(None, detector, self, identity=target_identity, identified=identified)
        image._loc = loc
        image.update(notify_target=notify_target)
        detector.sensor_settings.register_image(image)

        return image


    def range_to_identify(self, entity:core.SectorEntity, mass:Optional[float]=None, radius:Optional[float]=None) -> float:
        if mass is None:
            mass = config.Settings.generate.SectorEntities.asteroid.MASS
        if radius is None:
            radius = config.Settings.generate.SectorEntities.asteroid.RADIUS
        passive_profile = (config.Settings.sensors.COEFF_MASS * mass + config.Settings.sensors.COEFF_RADIUS * radius)

        threshold = self.compute_effective_threshold(entity)

        identification_range = np.sqrt(passive_profile / threshold * config.Settings.sensors.COEFF_IDENTIFICATION_FIDELITY / config.Settings.sensors.COEFF_DISTANCE)
        return identification_range

    def range_to_detect(self, entity:core.SectorEntity, mass:Optional[float]=None, radius:Optional[float]=None) -> float:
        if mass is None:
            mass = config.Settings.generate.SectorEntities.asteroid.MASS
        if radius is None:
            radius = config.Settings.generate.SectorEntities.asteroid.RADIUS
        passive_profile = (config.Settings.sensors.COEFF_MASS * mass + config.Settings.sensors.COEFF_RADIUS * radius)

        threshold = self.compute_effective_threshold(entity)

        detection_range = np.sqrt(passive_profile / threshold / config.Settings.sensors.COEFF_DISTANCE)
        return detection_range


    def sensor_ranges(self, entity:core.SectorEntity) -> tuple[float, float, float]:
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

    def profile_ranges(self, entity:core.SectorEntity) -> tuple[float, float]:
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

class SensorScanOrder(core.Order):
    """ Has the ship do a sensor scan and nothing else. """
    def __init__(self, *args:Any, images_ttl:float=0.5, sensor_power_ratio:float=1.0, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.images_ttl = images_ttl
        self.init_sensor_power_ratio = 0.0
        self.sensor_power_ratio = sensor_power_ratio
        self.images:Optional[Collection[core.AbstractSensorImage]] = None

    def _begin(self) -> None:
        self.init_sensor_power_ratio = self.ship.sensor_settings.sensor_power / self.ship.sensor_settings.max_sensor_power
        self.ship.sensor_settings.set_sensors(self.sensor_power_ratio)

    def _complete(self) -> None:
        self.ship.sensor_settings.set_sensors(self.init_sensor_power_ratio)

    def _cancel(self) -> None:
        self.ship.sensor_settings.set_sensors(self.init_sensor_power_ratio)

    def act(self, dt: float) -> None:
        if self.images is not None:
            self.complete_order()
        else:
            if not util.isclose(self.ship.sensor_settings.effective_sensor_power() / self.ship.sensor_settings.max_sensor_power, self.sensor_power_ratio):
                # wait until our sensors come up to the desired level
                self.gamestate.schedule_order(self.gamestate.timestamp+1.0, self)
            else:
                assert(self.ship.sector)
                # we need to keep these images alive for a few ticks
                self.images = list(self.ship.sector.sensor_manager.scan(self.ship))
                self.ship.sensor_settings.set_sensors(self.init_sensor_power_ratio)
                self.gamestate.schedule_order(self.gamestate.timestamp + self.images_ttl, self)

def pre_initialize(event_manager:events.EventManager) -> None:
    event_manager.register_events(Events, "sensors")
    event_manager.register_context_keys(ContextKeys, "sensors")
