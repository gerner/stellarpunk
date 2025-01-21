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

class ContextKeys(enum.IntEnum):
    DETECTOR = enum.auto()
    TARGET = enum.auto()
    SECTOR = enum.auto()
    STATIC_COUNT = enum.auto()
    DYNAMIC_COUNT = enum.auto()

class SensorImage(core.SectorEntityObserver, core.AbstractSensorImage):
    @classmethod
    def create_sensor_image(cls, target:Optional[core.SectorEntity], ship:core.SectorEntity, sensor_manager:core.AbstractSensorManager, identity:Optional[core.SensorIdentity]=None, identified:bool=False) -> "SensorImage":
        if identity is None:
            assert(target)
            identity = core.SensorIdentity(target)
        image = SensorImage(identity)
        image.set_sensor_manager(sensor_manager)
        image._target = target
        image._ship = ship
        image._detector_id = ship.entity_id
        image._identified = identified
        if target:
            target.observe(image)
        ship.observe(image)
        return image

    def __init__(self, identity:core.SensorIdentity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self._identity = identity
        self._sensor_manager:core.AbstractSensorManager = None # type: ignore
        self._ship:core.SectorEntity = None # type: ignore
        self._detector_id:uuid.UUID = None # type: ignore
        self._target:Optional[core.SectorEntity] = None
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
        return f'{self._identity.short_id} detected by {self._ship or self._detector_id} {self.age}s old'

    def __hash__(self) -> int:
        return hash((self._identity.entity_id, self._detector_id))

    # core.SectorEntityObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return self._detector_id

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self._target:
            # might be risky checking if detected on logically destroyed entity
            if self._ship and self._sensor_manager.detected(entity, self._ship):
                self._is_active = False
                self._inactive_reason = core.SensorImageInactiveReason.DESTROYED
            self._target = None
        elif entity == self._ship:
            if self._target:
                self._target.unobserve(self)
                self._target = None
            self._ship.unobserve(self)
            self._ship.sensor_settings.unregister_image(self)
            # we drop our _ship reference to let it get cleaned up
            self._ship = None # type: ignore
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
            # this sensor image is invalid if the detector migrates
            if self._target:
                self._target.unobserve(self)
                self._target = None
            self._ship.unobserve(self)
            self._ship.sensor_settings.unregister_image(self)
            self._ship = None # type: ignore
        else:
            raise ValueError(f'got entity_migrated for unexpected entity {entity}')


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
    def identified(self) -> bool:
        return self._identified

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

        if isinstance(self._target, sector_entity.Projectile):
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
        if self._ship and self._ship.sector and not self._target:
            # try to re-fetch the target if possible
            # this might happen if this image is created from intel
            # also possible if the target jumps away without us detecting them
            # and then comes back
            if self.identity.entity_id in self._ship.sector.entities:
                self._target = self._ship.sector.entities[self.identity.entity_id]
                self._target.observe(self)

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
                                    gamestate.event_manager.ck(ContextKeys.TARGET): self._target.short_id_int(),
                                }
                        )
                    # once identified, always identified
                    self._identified = True

                # let the target know they've been targeted by us
                if notify_target and self._sensor_manager.detected(self._ship, self._target):
                    #TODO: this is going to get triggered a lot. should we only
                    # do it the first time they get targeted?
                    #TODO: what about passengers? should they get this event too?
                    candidates = list(itertools.chain(core.crew(self._ship), core.crew(self._target)))
                    if candidates:
                        gamestate = core.Gamestate.gamestate
                        gamestate.trigger_event(
                                candidates,
                                gamestate.event_manager.e(Events.TARGETED),
                                {
                                    gamestate.event_manager.ck(ContextKeys.DETECTOR): self._ship.short_id_int(),
                                    gamestate.event_manager.ck(ContextKeys.TARGET): self._target.short_id_int(),
                                }
                        )

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
        if detector.sensor_settings.has_image(self.identity.entity_id):
            return detector.sensor_settings.get_image(self.identity.entity_id)

        identity = self._identity
        image = SensorImage.create_sensor_image(self._target, detector, self._sensor_manager, identity=self._identity, identified=self._identified)
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

        self._images:weakref.WeakValueDictionary[uuid.UUID, core.AbstractSensorImage] = weakref.WeakValueDictionary()

    def register_image(self, image:core.AbstractSensorImage) -> None:
        self._images[image.identity.entity_id] = image
    def unregister_image(self, image:core.AbstractSensorImage) -> None:
        del self._images[image.identity.entity_id]
    def has_image(self, target_id:uuid.UUID) -> bool:
        return target_id in self._images
    def get_image(self, target_id:uuid.UUID) -> core.AbstractSensorImage:
        return self._images[target_id]
    @property
    def images(self) -> Collection[core.AbstractSensorImage]:
        return list(self._images.values())

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
            image.initialize()
            image.update(notify_target=notify_target)
            detector.sensor_settings.register_image(image)
            return image

    def target_from_identity(self, target_identity:core.SensorIdentity, detector:core.SectorEntity, loc:npt.NDArray[np.float64], notify_target:bool=True, identified:bool=True) -> core.AbstractSensorImage:
        if detector.sensor_settings.has_image(target_identity.entity_id):
            return detector.sensor_settings.get_image(target_identity.entity_id)

        image = SensorImage.create_sensor_image(None, detector, self, identity=target_identity, identified=identified)
        image._loc = loc
        detector.sensor_settings.register_image(image)

        return image


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
    def __init__(self, *args:Any, images_ttl:float=0.5, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.images:Optional[Collection[core.AbstractSensorImage]] = None
        self.images_ttl = images_ttl

    def act(self, dt: float) -> None:
        if self.images is not None:
            self.complete_order()
        else:
            assert(self.ship.sector)
            # we need to keep these images alive for a few ticks
            self.images = list(self.ship.sector.sensor_manager.scan(self.ship))
            self.gamestate.schedule_order(self.gamestate.timestamp + self.images_ttl, self)

def pre_initialize(event_manager:events.EventManager) -> None:
    event_manager.register_events(Events, "sensors")
    event_manager.register_context_keys(ContextKeys, "sensors")
