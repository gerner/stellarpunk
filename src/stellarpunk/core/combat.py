""" Combat """

import logging
import enum
import collections
import math
import uuid
from typing import Optional, Tuple, Any, Set, List, Deque, Iterator, MutableMapping, Type

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, core, config
from stellarpunk.orders import movement, collision

logger = logging.getLogger(__name__)

#TODO: can we just globally assign this? how do we keep this in sync with others?
POINT_DEFENSE_COLLISION_TYPE = 1

def initialize_gamestate(gamestate:core.Gamestate) -> None:
    logger.info("initializing combat for this gamestate...")
    for sector in gamestate.sectors.values():
        sector.space.add_collision_handler(POINT_DEFENSE_COLLISION_TYPE, core.SECTOR_ENTITY_COLLISION_TYPE, pre_solve = point_defense_collision_handler)

def point_defense_collision_handler(space:cymunk.Space, arbiter:cymunk.Arbiter) -> bool:
    # shape_a is the point defense shape, shape_b is the sector entity colliding
    (shape_a, shape_b) = arbiter.shapes
    point_defense_effect:PointDefenseEffect = shape_a.body.data
    entity:core.SectorEntity = shape_b.body.data
    point_defense_effect.add_collision(entity)

    return False

def is_hostile(craft:core.SectorEntity, threat:core.SectorEntity) -> bool:
    #TODO: improve this logic
    hostile = False
    # is it a weapon (e.g. missile)
    if isinstance(threat, Missile):
        hostile = True
    # is it known to be hostile?
    # is it running with its transponder off?
    if not threat.sensor_settings.transponder:
        hostile = True
    # is it otherwise operating in a hostile manner (e.g. firing weapons)
    return hostile

def damage(craft:core.SectorEntity) -> bool:
    """ damages craft, returning true iff craft is destroyed. """
    # for now we just destroy the entity
    core.Gamestate.gamestate.destroy_entity(craft)
    return True

class Missile(core.Ship):
    id_prefix = "MSL"

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # missiles don't run transponders
        self.firer:Optional[core.SectorEntity] = None

class TimedOrderTask(core.ScheduledTask, core.OrderObserver):
    @staticmethod
    def ttl_order(order:core.Order, ttl:float) -> "TimedOrderTask":
        tot = TimedOrderTask(order)
        core.Gamestate.gamestate.schedule_task(core.Gamestate.gamestate.timestamp + ttl, tot)
        return tot
    def __init__(self, order:core.Order) -> None:
        self.order = order
        order.observe(self)
    def order_cancel(self, order:core.Order) -> None:
        core.Gamestate.gamestate.unschedule_task(self)
    def order_complete(self, order:core.Order) -> None:
        core.Gamestate.gamestate.unschedule_task(self)
    def act(self) -> None:
        self.order.cancel_order()

class ThreatTracker(core.SectorEntityObserver):
    def __init__(self, craft:core.SectorEntity, threat_ttl:float=120.) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.craft:core.SectorEntity = craft
        self.threats:Set[core.AbstractSensorImage] = set()
        self.threat_ids:Set[uuid.UUID] = set()
        self.last_target_ts = -np.inf
        self.closest_threat:Optional[core.AbstractSensorImage] = None

        self.threat_ttl = threat_ttl

    def __len__(self) -> int:
        return len(self.threats)

    def __iter__(self) -> Iterator[core.AbstractSensorImage]:
        for x in self.threats:
            yield x

    def start_tracking(self) -> None:
        self.craft.observe(self)
        self.last_target_ts = -np.inf

    def stop_tracking(self) -> None:
        self.craft.unobserve(self)

    def add_threat(self, threat:core.AbstractSensorImage) -> None:
        if threat.identity.entity_id in self.threat_ids:
            return
        self.logger.debug(f'adding threat {threat}')
        self.threats.add(threat)
        self.threat_ids.add(threat.identity.entity_id)

    def entity_targeted(self, craft:core.SectorEntity, threat:core.SectorEntity) -> None:
        assert craft == self.craft
        assert self.craft.sector
        if threat.entity_id in self.threat_ids:
            self.last_target_ts = core.Gamestate.gamestate.timestamp
            return

        # no need to worry about non-hostile threats
        if not is_hostile(self.craft, threat):
            return

        self.last_target_ts = core.Gamestate.gamestate.timestamp
        self.add_threat(self.craft.sector.sensor_manager.target(threat, self.craft))

    def update_threats(self) -> Optional[core.AbstractSensorImage]:
        # first remove eliminated/expired threats
        closest_dist = np.inf
        last_closest_threat = self.closest_threat
        self.closest_threat = None
        dead_threats:List[core.AbstractSensorImage] = []
        for t in self.threats:
            t.update()
            if not t.is_active() or t.age > self.threat_ttl:
                self.logger.debug(f'dropping threat {t}')
                dead_threats.append(t)
            else:
                dist = util.distance(t.loc, self.craft.loc)
                if dist < closest_dist:
                    self.closest_threat = t
                    closest_dist = dist
        for t in dead_threats:
            self.threats.remove(t)
            self.threat_ids.remove(t.identity.entity_id)

        return self.closest_threat

class MissileOrder(movement.PursueOrder, core.CollisionObserver):
    """ Steer toward a collision with the target """

    @staticmethod
    def spawn_missile(ship:core.Ship, gamestate:core.Gamestate, target_image:core.AbstractSensorImage, initial_velocity:float=100, spawn_distance_forward:float=100, spawn_radius:float=100, owner:Optional[core.SectorEntity]=None) -> Missile:
        assert ship.sector
        loc = gamestate.generator.gen_sector_location(ship.sector, center=ship.loc + util.polar_to_cartesian(spawn_distance_forward, ship.angle), occupied_radius=75, radius=spawn_radius, strict=True)
        v = util.polar_to_cartesian(initial_velocity, ship.angle) + ship.velocity
        new_entity = gamestate.generator.spawn_sector_entity(Missile, ship.sector, loc[0], loc[1], v=v, w=0.0)
        assert isinstance(new_entity, Missile)
        missile:Missile = new_entity
        missile.firer = owner
        missile_order = MissileOrder.create_missile_order(target_image.copy(missile), missile, gamestate)
        missile.prepend_order(missile_order)
        missile.sensor_settings.set_sensors(1.0)

        return missile

    @classmethod
    def create_missile_order[T:"MissileOrder"](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        return cls.create_pursue_order(*args, **kwargs)

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        kwargs["avoid_collisions"] = False
        super().__init__(*args, **kwargs)
        self.avoid_collisions=False

        self.ttl = 240
        self.expiration_time = self.gamestate.timestamp + self.ttl

    def _begin(self) -> None:
        assert self.ship.sector
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.ship.sector.register_collision_observer(self.ship.entity_id, self)

    def _complete(self) -> None:
        super()._complete()
        # ship might already have been removed from the sector
        # assume if that happens it got unregistered
        if self.ship.sector:
            self.logger.debug(f'completing missile after {self.gamestate.timestamp - self.started_at}s at distance {util.distance(self.ship.loc, self.target.loc)}m age {self.target.age}s')
            self.ship.sector.unregister_collision_observer(self.ship.entity_id, self)
        else:
            self.logger.debug(f'completing missile after {self.gamestate.timestamp - self.started_at}s age {self.target.age}s')
        self.gamestate.destroy_entity(self.ship)

    def _cancel(self) -> None:
        self._complete()

    # core.CollisionObserver
    def collision(self, missile:core.SectorEntity, target:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        assert missile == self.ship
        if target.entity_id == self.target.identity.entity_id:
            self.logger.debug(f'missile hit desired target {target} impulse: {impulse} ke: {ke}!')
        else:
            self.logger.debug(f'missile hit collateral target {target} impulse: {impulse} ke: {ke}!')

        self.complete_order()
        damage(target)

    def _is_complete(self) -> bool:
        return self.gamestate.timestamp > self.expiration_time or not self.target.is_active()

    def act(self, dt:float) -> None:
        super().act(dt)

class AttackOrder(movement.AbstractSteeringOrder):
    """ Objective is to destroy a target. """

    class State(enum.Enum):
        APPROACH = enum.auto()
        WITHDRAW = enum.auto()
        SHADOW = enum.auto()
        FIRE = enum.auto()
        AIM = enum.auto()
        LAST_LOCATION = enum.auto()
        SEARCH = enum.auto()
        GIVEUP = enum.auto()

    @classmethod
    def create_attack_order[T:"AttackOrder"](cls:Type[T], target:core.AbstractSensorImage, *args:Any, distance_min:float=7.5e4, distance_max:float=2e5, max_active_age:float=35, max_passive_age:float=15, search_distance:float=2.5e4, max_fire_rel_bearing:float=np.pi/8, max_missiles:Optional[int]=None, **kwargs:Any) -> T:
        o = cls.create_abstract_steering_order(*args, distance_min=distance_min, distance_max=distance_max, max_active_age=max_active_age, max_passive_age=max_passive_age, search_distance=search_distance, max_fire_rel_bearing=max_fire_rel_bearing, max_missiles=max_missiles, **kwargs)
        assert o.ship.sector
        o.target = target
        return o

    def __init__(self, *args:Any, distance_min:float=7.5e4, distance_max:float=2e5, max_active_age:float=35, max_passive_age:float=15, search_distance:float=2.5e4, max_fire_rel_bearing:float=np.pi/8, max_missiles:Optional[int]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.target:core.AbstractSensorImage = None # type: ignore
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.max_active_age = max_active_age
        self.max_passive_age = max_passive_age
        self.search_distance = search_distance
        self.max_fire_age = 3.0
        self.min_profile_to_threshold = 4.0
        self.fire_backoff_time = 1.0
        self.max_fire_rel_bearing = max_fire_rel_bearing
        self.ttl_order_time = 2.
        #TODO: a temporary hack to limit how many missiles we can shoot
        self.max_missiles = max_missiles

        self.state = AttackOrder.State.APPROACH

        self.last_fire_ts = -3600.
        #TODO: fire period is a temporary hack to limit firing
        self.fire_period = 15.
        self.missiles_fired = 0

    def __str__(self) -> str:
        return f'Attack: {self.target.identity.short_id} state: {self.state} age: {self.target.age:.1f}s dist: {util.human_distance(float(np.linalg.norm(self.target.loc-self.ship.loc)))}'

    def _begin(self) -> None:
        super()._begin()
        self.logger.debug(f'beginning attack on {self.target.identity.short_id}')

    def _is_complete(self) -> bool:
        return not self.target.is_active()

    def _ttl_order(self, order:core.Order, ttl:Optional[float]=None) -> None:
        if ttl is None:
            ttl = self.ttl_order_time
        TimedOrderTask.ttl_order(order, ttl)
        self._add_child(order)

    def _do_last_location(self) -> None:
        self._ttl_order(movement.PursueOrder.create_pursue_order(self.target, self.ship, self.gamestate, arrival_distance=self.search_distance*0.8, max_speed=self.ship.max_speed()*5, final_speed=self.ship.max_thrust / self.ship.mass * 5.))

    def _do_search(self) -> None:
        #TODO: search, for now give up
        self.logger.debug(f'giving up search for target {self.target.identity.short_id}')
        self.state = AttackOrder.State.GIVEUP

    def _do_approach(self) -> None:
        self._ttl_order(movement.PursueOrder.create_pursue_order(self.target, self.ship, self.gamestate, arrival_distance=self.distance_max-0.2*(self.distance_max-self.distance_min), max_speed=self.ship.max_speed()*2.))

    def _do_withdraw(self) -> None:
        self._ttl_order(movement.EvadeOrder.create_evade_order(self.target, self.ship, self.gamestate, escape_distance=self.distance_min+0.2*(self.distance_max-self.distance_min)))

    def _do_shadow(self, dt:float) -> None:
        assert self.ship.sector
        # make velocity parallel to target (i.e. want relative velocity to be zero)
        target_velocity = self.target.velocity

        # add velocity component to get to center of standoff range
        #away_vector = self.ship.loc - self.target.loc
        #dist = util.magnitude(*away_vector)
        #away_vector = away_vector / dist
        #dist = (self.distance_max + self.distance_min) / 2 - dist
        #away_vector *= dist / 5
        #mag = util.magnitude(*away_vector)
        #away_vector = away_vector / mag * min(mag, 0.2 * util.magnitude(*target_velocity))
        #target_velocity += away_vector

        # avoid collisions
        collision_dv, approach_time = self._avoid_collisions_dv(
                self.ship.sector,
                desired_direction=cymunk.Vec2d(target_velocity))

        if not util.both_almost_zero(collision_dv):
            target_velocity = self.ship.velocity + collision_dv

        # TODO: choose a max thrust appropriate for desired sensor profile

        shadow_time = collision.accelerate_to(self.ship.phys, cymunk.Vec2d(target_velocity), dt, self.ship.max_speed(), self.ship.max_torque, self.ship.max_thrust, self.ship.max_fine_thrust, self.ship.sensor_settings)
        self.gamestate.schedule_order(self.gamestate.timestamp + min(shadow_time, 1/10), self)

    def _do_fire(self) -> None:
        rel_bearing = util.normalize_angle(util.bearing(self.ship.loc, self.target.loc) - self.ship.angle, shortest=True)
        missile = MissileOrder.spawn_missile(self.ship, self.gamestate, self.target, owner=self.ship)
        self.logger.debug(f'firing missile {missile} at distance {util.distance(self.ship.loc, self.target.loc)}m {rel_bearing=} with ptr {self.target.profile / self.ship.sensor_settings.max_threshold}')

        assert util.distance(missile.loc, self.target.loc) < util.distance(self.ship.loc, self.target.loc)
        self.missiles_fired += 1
        self.last_fire_ts = self.gamestate.timestamp

        #TODO: after firing what should we do?
        self._ttl_order(core.NullOrder.create_null_order(self.ship, self.gamestate), ttl=self.fire_backoff_time)

    def _do_aim(self, dt:float) -> None:
        # get in close enough that a launched missile will be able to lock on to the target

        if self.target.profile / self.ship.sensor_settings.max_threshold < self.min_profile_to_threshold:
            self._ttl_order(movement.PursueOrder.create_pursue_order(self.target, self.ship, self.gamestate, arrival_distance=1e3, max_speed=self.ship.max_speed()*5, final_speed=self.ship.max_thrust / self.ship.mass * 5.))
            return

        bearing = util.bearing(self.ship.loc, self.target.loc)
        rotate_time = collision.rotate_to(self.ship.phys, bearing, dt, self.ship.max_torque)
        self.gamestate.schedule_order(self.gamestate.timestamp + min(rotate_time, 1/10), self)

    def _set_sensors(self) -> None:
        assert self.ship.sector
        if self.target.age > self.max_passive_age:
            self.ship.sensor_settings.set_sensors(1.0)
        else:
            self.ship.sensor_settings.set_sensors(0.0)

    def _choose_state(self) -> "AttackOrder.State":
        if self.gamestate.timestamp - self.last_fire_ts < self.fire_backoff_time:
            # back off immediately following firing a missile to reduce risk of
            # colliding with our own missile
            return AttackOrder.State.WITHDRAW

        distance = np.linalg.norm(self.target.loc - self.ship.loc)
        if self.target.age > self.max_active_age:
            if distance > self.search_distance:
                return AttackOrder.State.LAST_LOCATION
            else:
                return AttackOrder.State.SEARCH

        if distance > self.distance_max:
            return AttackOrder.State.APPROACH
        #TODO: more complex logic around firing weapons, also some limit on
        # weapon use
        # if weapon systems not ready, ready weapons
        # if we've got enough confidence take shot, else gain confidence
        # only fire a missile if we're vaguely pointed toward the target
        if self.target.age < self.max_fire_age and self.gamestate.timestamp - self.last_fire_ts > self.fire_period:
            #TODO: temporary hack to limit attacking
            if self.max_missiles is not None and self.missiles_fired >= self.max_missiles:
                return AttackOrder.State.GIVEUP
            if abs(util.normalize_angle(util.bearing(self.ship.loc, self.target.loc) - self.ship.angle, shortest=True)) < self.max_fire_rel_bearing and self.target.profile / self.ship.sensor_settings.max_threshold > self.min_profile_to_threshold:
                return AttackOrder.State.FIRE
            else:
                return AttackOrder.State.AIM

        if distance < self.distance_min:
            return AttackOrder.State.WITHDRAW
        else:
            return AttackOrder.State.SHADOW

    def act(self, dt:float) -> None:
        assert self.ship.sector
        self.target.update()

        self._set_sensors()

        self.state = self._choose_state()

        if self.state == AttackOrder.State.LAST_LOCATION:
            self._do_last_location()
        elif self.state == AttackOrder.State.SEARCH:
            self._do_search()
        elif self.state == AttackOrder.State.APPROACH:
            self._do_approach()
        elif self.state == AttackOrder.State.WITHDRAW:
            self._do_withdraw()
        elif self.state == AttackOrder.State.SHADOW:
            self._do_shadow(dt)
        elif self.state == AttackOrder.State.FIRE:
            self._do_fire()
        elif self.state == AttackOrder.State.AIM:
            self._do_aim(dt)
        elif self.state == AttackOrder.State.GIVEUP:
            self.cancel_order()
        else:
            raise ValueError(f'unknown attack order state {self.state}')

class PDTarget:
    """ Tracks objects in point defense firing cone. """
    def __init__(self, entity:core.SectorEntity):
        self.entity = entity
        self.birth_ts = core.Gamestate.gamestate.timestamp
        self.last_seen = self.birth_ts
        self.last_roll = -np.inf

    def __str__(self) -> str:
        return f'PDTarget {self.entity.short_id()} {self.age:0.2f}s old sls:{self.since_last_seen:0.2f} slr:{self.since_last_roll:0.2f}'

    @property
    def age(self) -> float:
        return core.Gamestate.gamestate.timestamp - self.birth_ts

    @property
    def since_last_seen(self) -> float:
        return core.Gamestate.gamestate.timestamp - self.last_seen

    @property
    def since_last_roll(self) -> float:
        return core.Gamestate.gamestate.timestamp - self.last_roll

# TODO: should point defense really be creating all these sector entities?
#   perhaps a better choice would be to create a cone shaped area to monitor
#   inside that space we can abstractly model point defense working
class PointDefenseEffect(core.Effect, core.SectorEntityObserver):
    class State(enum.Enum):
        IDLE = enum.auto()
        ACTIVE = enum.auto()

    def __init__(self, craft:core.SectorEntity, *args:Any, threat_tracker:Optional[ThreatTracker]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.craft = craft
        if threat_tracker is None:
            self.threat_tracker = ThreatTracker(craft)
            self.own_tracker = True
        else:
            self.threat_tracker = threat_tracker
            self.own_tracker = False
        self.state = PointDefenseEffect.State.IDLE
        self.current_target:Optional[core.AbstractSensorImage] = None
        self.targets_destroyed = 0

        # how long between checks for new target if we have none
        self.idle_interval = 3.0
        self.active_interval = 0.3
        self.pdtarget_expiration = 0.5

        # phalanx has rof of 4500/min = 75/sec, muzzle velocity of 1100 m/s
        self.muzzle_velocity = 3000.
        self.projectile_ttl = 4.0
        self.cone_half_angle = np.pi/4./2.

        # pd sensor cone
        self._pd_shape:Optional[cymunk.Shape] = None
        # constraint that translates the sensor cone with the craft
        self._pd_shape_constraint:Optional[cymunk.Constraint] = None
        # track objects in our firing cone and when they enter the cone
        self._pd_collisions:MutableMapping[uuid.UUID, PDTarget] = {}

        # tracks if we saw an entity in our cone since the last time we acted
        self._collided = False

    def _begin(self) -> None:
        if self.own_tracker:
            self.threat_tracker.start_tracking()
        self.state = PointDefenseEffect.State.IDLE
        self.craft.observe(self)
        self.gamestate.schedule_effect_immediate(self, jitter=0.1)

    def _complete(self) -> None:
        if self.state == PointDefenseEffect.State.ACTIVE:
            self._deactivate()
        if self.own_tracker:
            self.threat_tracker.stop_tracking()
        self.craft.unobserve(self)

    def _cancel(self) -> None:
        self._complete()

    def _activate(self) -> None:
        """ IDLE -> ACTIVE transition logic """
        self.logger.info("activate")
        self.state = PointDefenseEffect.State.ACTIVE

        # create a cone centered on us that moves with us, but stays pointed in
        # a set direction. we'll aim the cone separately.

        # the body we're using to model point defense needs to be non-static
        # so it must have non-zero mass/moment. we make it small to minimize
        # impact on the ship, although technically the ship will perform
        # slightly worse while point defense is on since it's dragging around
        # this small body.
        # an alternative would be to add the pd sensor cone to the ship's body
        # but then we need to destroy it and recreate it to aim it and it would
        # rotate as the ship rotates instead of staying fixed on a heading
        mass = 0.1
        radius = 0.1
        moment = cymunk.moment_for_circle(mass, 0, radius)
        body = cymunk.Body(mass, moment)
        offset_x = self.muzzle_velocity*self.projectile_ttl
        offset_y = np.tan(self.cone_half_angle)*offset_x#np.tan(self.dispersion_angle/2)*offset_x
        self._pd_shape = cymunk.Poly(body, [(0.0, 0.0), (offset_x, offset_y), (offset_x, -offset_y)])
        self._pd_shape.collision_type = POINT_DEFENSE_COLLISION_TYPE
        self._pd_shape.group = id(self.craft.phys)
        self._pd_shape.sensor = True
        body.position = self.craft.phys.position
        body.data = self
        self.sector.space.add(self._pd_shape.body, self._pd_shape)
        body.angle = self.craft.angle
        self._pd_shape_constraint = cymunk.PivotJoint(self.craft.phys, self._pd_shape.body, self.craft.phys.position)
        self.sector.space.add(self._pd_shape_constraint)

    def _deactivate(self) -> None:
        """ ACTIVE -> IDLE transition logic """
        self.logger.info("deactivate")
        assert self._pd_shape
        self.state = PointDefenseEffect.State.IDLE

        # remove point defense cone.
        self.sector.space.remove(self._pd_shape_constraint, self._pd_shape.body, self._pd_shape)
        self._pd_shape_constraint = None
        self._pd_shape = None
        self._pd_collisions.clear()

    def add_collision(self, entity:core.SectorEntity) -> None:
        assert self.state == PointDefenseEffect.State.ACTIVE
        logger.info(f'point defense cone collision with {entity}')
        if entity.entity_id not in self._pd_collisions:
            self._pd_collisions[entity.entity_id] = PDTarget(entity)
        else:
            self._pd_collisions[entity.entity_id].last_seen = core.Gamestate.gamestate.timestamp

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if self.craft == entity:
            self.cancel_effect()
        else:
            raise ValueError(f'got unexpected entity {entity}')

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if self.craft == entity:
            self.craft.unobserve(self)
            self.cancel_effect()
        else:
            raise ValueError(f'got unexpected entity {entity}')

    def bbox(self) -> Tuple[float, float, float, float]:
        return (0., 0., 0., 0.)

    def act(self, dt:float) -> None:
        self._collided = False
        if self.is_complete():
            self.logger.debug(f'point defense is complete')
            self.cancel_effect()
            return

        if self.own_tracker:
            self.threat_tracker.update_threats()
        target = self.threat_tracker.closest_threat
        if target is None:
            if self.state == PointDefenseEffect.State.ACTIVE:
                self._deactivate()
            self.gamestate.schedule_effect(core.Gamestate.gamestate.timestamp + self.idle_interval, self)
            return
        if self.state == PointDefenseEffect.State.IDLE:
            self._activate()

        self._do_point_defense(target)

    def _handle_collision(self, target:PDTarget) -> bool:
        if target.since_last_roll < config.Settings.combat.point_defense.ROLL_INTERVAL:
            return False
        if target.entity.entity_id not in self.threat_tracker.threat_ids:
            p = config.Settings.combat.point_defense.COLLATERAL_HIT_PROBABILITY
        else:
            p = config.Settings.combat.point_defense.THREAT_HIT_PROBABILITY
        roll = self.gamestate.random.uniform()
        if roll < p:
            self.logger.info(f'pd hit {target.entity.entity_id} {roll} < {p}')
            if damage(target.entity):
                self.targets_destroyed += 1
                return True
        else:
            self.logger.info(f'pd miss {target.entity.entity_id} {roll} > {p}')
        target.last_roll = core.Gamestate.gamestate.timestamp
        return False

    def _do_point_defense(self, target:core.AbstractSensorImage) -> None:
        self.logger.info(f'pd {target}')
        assert self._pd_shape
        # handle any collisions with the cone shape
        remove_ids:List[uuid.UUID] = []
        for entity_id, pdtarget in self._pd_collisions.items():
            self.logger.info(f'pdtarget {pdtarget}')
            if pdtarget.since_last_seen > self.pdtarget_expiration:
                remove_ids.append(entity_id)
            elif util.isclose(pdtarget.since_last_seen, 0.0):
                if self._handle_collision(pdtarget):
                    remove_ids.append(entity_id)
        for entity_id in remove_ids:
            del self._pd_collisions[entity_id]

        # aim point defense cone shape toward target
        self._pd_shape.body.angle = util.bearing(self.craft.loc, target.loc)
        self.current_target = target
        self.gamestate.schedule_effect(core.Gamestate.gamestate.timestamp + self.active_interval, self)


class HuntOrder(core.Order):
    @classmethod
    def create_hunt_order[T:"HuntOrder"](cls:Type[T], target_id:uuid.UUID, *args:Any, start_loc:Optional[npt.NDArray[np.float64]], **kwargs:Any) -> T:
        o =  cls.create_order(*args, target_id, **kwargs)
        if start_loc is None:
            o.start_loc = o.ship.loc
        else:
            o.start_loc = start_loc
        return o

    def __init__(self, target_id:uuid.UUID, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_id = target_id
        self.attack_order:Optional[AttackOrder] = None
        self.start_loc:npt.NDArray[np.float64] = None # type: ignore
        self.ttl_order_time = 5.

    def _ttl_order(self, order:core.Order, ttl:Optional[float]=None) -> None:
        if ttl is None:
            ttl = self.ttl_order_time
        TimedOrderTask.ttl_order(order, ttl)
        self._add_child(order)

    def _scan_target(self) -> Optional[core.AbstractSensorImage]:
        assert self.ship.sector
        # note: the model is that this ship doesn't know if the target is in
        # sector or not. we're reaching inside sector.entities for convenience
        if self.target_id in self.ship.sector.entities:
            target = self.ship.sector.entities[self.target_id]
            if self.ship.sector.sensor_manager.detected(target, self.ship):
                return self.ship.sector.sensor_manager.target(target, self.ship)
        return None

    #TODO: is_complete?

    def _begin(self) -> None:
        self.ship.sensor_settings.set_transponder(False)

    def _complete(self) -> None:
        self.ship.sensor_settings.set_transponder(True)

    def _cancel(self) -> None:
        self.ship.sensor_settings.set_transponder(True)

    def act(self, dt:float) -> None:
        # alternate between traveling to a search point and scanning for the
        # target
        target = self._scan_target()
        if target:
            self.attack_order = AttackOrder.create_attack_order(target, self.ship, self.gamestate)
            self._add_child(self.attack_order)
        else:
            # choose a search location
            loc = self.start_loc
            # go there for a bit
            self._ttl_order(movement.GoToLocation.create_go_to_location(loc, self.ship, self.gamestate))

class FleeOrder(core.Order, core.SectorEntityObserver):
    """ Keeps track of threats and flees from them until "safe" """
    @classmethod
    def create_flee_order[T:"FleeOrder"](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        o = cls.create_order(*args, **kwargs)
        assert o.ship.sector
        o.threat_tracker = ThreatTracker(o.ship)
        o.point_defense = PointDefenseEffect(o.ship, o.ship.sector, o.gamestate, threat_tracker=o.threat_tracker)
        o.max_thrust = o.ship.max_thrust
        return o

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.ttl_order_time = 5.
        self.last_target_ttl = 30.

        self.threat_tracker:ThreatTracker = None # type: ignore
        self.point_defense:PointDefenseEffect = None # type: ignore
        self.max_thrust = 0.0

    def _ttl_order(self, order:core.Order) -> None:
        TimedOrderTask.ttl_order(order, self.ttl_order_time)
        self._add_child(order)

    def add_threat(self, threat:core.AbstractSensorImage) -> None:
        self.threat_tracker.add_threat(threat)

    def _begin(self) -> None:
        assert self.ship.sector
        self.ship.sensor_settings.set_sensors(0.0)
        self.ship.sensor_settings.set_transponder(False)
        self.threat_tracker.start_tracking()
        self.ship.sector.add_effect(self.point_defense)

    def _complete(self) -> None:
        self.point_defense.cancel_effect()
        self.ship.sensor_settings.set_sensors(1.0)
        self.ship.sensor_settings.set_transponder(True)
        self.threat_tracker.stop_tracking()

    def _cancel(self) -> None:
        self.ship.sensor_settings.set_sensors(1.0)
        self.ship.sensor_settings.set_transponder(True)
        self.threat_tracker.stop_tracking()

    def _is_complete(self) -> bool:
        if len(self.threat_tracker) == 0:
            return True
        if self.gamestate.timestamp - self.threat_tracker.last_target_ts > self.last_target_ttl:
            return True
        return False

    def _choose_thrust(self) -> float:
        assert self.ship.sector
        # determine thrust we want to use to evade
        # if we've got low pressure to get away fast, we want to minimize
        # thrust in order to minimize our sensor profile

        # ideally we'd choose a thrust that keeps us hidden under active
        # sensors but at least we should choose a thrust that requires active
        # sensors
        max_active_thrust = self.ship.max_thrust
        max_passive_thrust = self.ship.max_thrust
        for threat in self.threat_tracker:
            dist_sq = util.distance_sq(self.ship.loc, threat.loc)
            active_threshold_thrust = self.ship.sector.sensor_manager.compute_thrust_for_profile(self.ship, dist_sq, self.ship.sensor_settings.effective_threshold(self.ship.sensor_settings.max_sensor_power))
            passive_threshold_thrust = self.ship.sector.sensor_manager.compute_thrust_for_sensor_power(self.ship, dist_sq, 0.0)
            if active_threshold_thrust < max_active_thrust:
                max_active_thrust = active_threshold_thrust
            if passive_threshold_thrust < max_passive_thrust:
                max_passive_thrust = passive_threshold_thrust

        if max_active_thrust > 0.:
            return max_active_thrust * 0.8
        elif max_passive_thrust > 0.:
            return max_passive_thrust * 0.8
        else:
            return self.ship.max_thrust

    def act(self, dt:float) -> None:
        self.logger.debug(f'act at {self.gamestate.timestamp}')
        self.threat_tracker.update_threats()
        if self.threat_tracker.closest_threat is None:
            self.logger.debug(f'no more threats, completing flee order')
            self.complete_order()
            return

        #TODO: better logic on how to evade all the threats simultaneously
        # evade closest threat

        self.max_thrust = self._choose_thrust()
        self._ttl_order(movement.EvadeOrder.create_evade_order(self.threat_tracker.closest_threat, self.ship, self.gamestate, max_thrust=self.max_thrust))
