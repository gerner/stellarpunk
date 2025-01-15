""" Orders that can be given to ships. """

from __future__ import annotations

import math
import uuid
from typing import Optional, Any, Type

import numpy as np
import numpy.typing as npt

from stellarpunk import util, core, effects, econ, events, sensors
from stellarpunk.core import sector_entity
from stellarpunk.narrative import director
from . import movement

from .movement import GoToLocation, RotateOrder
from .steering import ZERO_VECTOR

class MineOrder(core.OrderObserver, core.EffectObserver, core.Order):
    @classmethod
    def create_mine_order[T:"MineOrder"](cls:Type[T], target: sector_entity.Asteroid, amount: float, *args: Any, max_dist:float=2e3, **kwargs: Any) -> MineOrder:
        o = cls.create_order(*args, max_dist=max_dist, **kwargs)
        o.target = target
        o.eow = core.EntityOrderWatch(o, target)
        o.amount = min(amount, o.ship.cargo_capacity)
        return o

    def __init__(self, *args: Any, max_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target:core.Asteroid = None # type: ignore
        self.eow:core.EntityOrderWatch = None # type: ignore
        self.max_dist = max_dist
        self.amount = 0.0
        self.mining_effect:Optional[effects.MiningEffect] = None
        self.mining_rate = 2e1

    def estimate_eta(self) -> float:
        docking_eta = DockingOrder.compute_eta(self.ship, self.target)
        if docking_eta > 0:
            return docking_eta + 5 + self.amount / self.mining_rate
        elif self.mining_effect is not None:
            return self.mining_effect.estimate_eta()
        else:
            raise Exception("no docking time, but also no mining effect")


    def _begin(self) -> None:
        self.init_eta = (
                DockingOrder.compute_eta(self.ship, self.target)
                + 5
                + self.amount / self.mining_rate
        )

    def _cancel(self) -> None:
        if self.mining_effect:
            self.mining_effect.cancel_effect()
            self.mining_effect.cancel_effect()

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def effect_complete(self, effect:core.Effect) -> None:
        assert effect == self.mining_effect
        if self.completed_at > 0:
            return
        self.gamestate.schedule_order_immediate(self)

    def effect_cancel(self, effect:core.Effect) -> None:
        assert effect == self.mining_effect
        if self.completed_at > 0:
            return
        self.gamestate.schedule_order_immediate(self)

    def order_complete(self, order:core.Order) -> None:
        if self.completed_at > 0:
            return
        self.gamestate.schedule_order_immediate(self)

    def order_cancel(self, order:core.Order) -> None:
        if self.completed_at > 0:
            return
        self.gamestate.schedule_order_immediate(self)

    def _is_complete(self) -> bool:
        return (self.mining_effect is not None and self.mining_effect.is_complete())

    def act(self, dt: float) -> None:
        if self.ship.sector != self.target.sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of target {self.target.sector}')
        # grab resources from the asteroid and add to our cargo
        distance = util.distance(self.ship.loc,self.target.loc) - self.target.radius
        if distance > self.max_dist:
            order = DockingOrder.create_docking_order(self.target, self.ship, self.gamestate, surface_distance=self.max_dist, observer=self)
            self._add_child(order)

        elif movement.KillVelocityOrder.in_motion(self.ship):
            self._add_child(movement.KillVelocityOrder.create_kill_velocity_order(self.ship, self.gamestate, observer=self))
        elif not self.mining_effect:
            assert self.ship.phys.torque == 0.
            assert self.ship.sector is not None
            self.mining_effect = effects.MiningEffect.create_transfer_cargo_effect(
                    self.target.resource, self.amount, self.target, self.ship, self.ship.sector, self.gamestate, transfer_rate=self.mining_rate, observer=self)
            self.ship.sector.add_effect(self.mining_effect)

        # else wait for the mining effect

class TransferCargo(core.OrderObserver, core.EffectObserver, core.Order):
    @classmethod
    def transfer_rate(cls) -> float:
        return 1e2

    @classmethod
    def create_transfer_cargo[T:"TransferCargo"](cls:Type[T], target: core.SectorEntity, resource: int, amount: float, *args:Any, max_dist:float=2e3, **kwargs:Any) -> T:
        o = cls.create_order(*args, resource, amount, **kwargs)
        o.target = target
        o.eow = core.EntityOrderWatch(o, target)
        return o

    def __init__(self, resource: int, amount: float, *args: Any, max_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.target:core.SectorEntity = None # type: ignore
        self.eow:core.EntityOrderWatch = None # type: ignore
        self.resource = resource
        self.amount = amount
        self.transferred = 0.
        self.max_dist = max_dist

        self.transfer_effect:Optional[core.Effect] = None

    def _begin(self) -> None:
        self.init_eta = (
                DockingOrder.compute_eta(self.ship, self.target)
                + self.amount / self.transfer_rate()
        )

    def _cancel(self) -> None:
        if self.transfer_effect:
            self.transfer_effect.cancel_effect()

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def effect_complete(self, effect:core.Effect) -> None:
        assert effect == self.transfer_effect
        self.gamestate.schedule_order_immediate(self)

    def effect_cancel(self, effect:core.Effect) -> None:
        assert effect == self.transfer_effect
        self.gamestate.schedule_order_immediate(self)

    def order_complete(self, order:core.Order) -> None:
        self.gamestate.schedule_order_immediate(self)

    def order_cancel(self, order:core.Order) -> None:
        self.gamestate.schedule_order_immediate(self)

    def _is_complete(self) -> bool:
        return self.transfer_effect is not None and self.transfer_effect.is_complete()

    def act(self, dt:float) -> None:
        if self.ship.sector is None or self.ship.sector != self.target.sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of target {self.target.sector}')
        # if we're too far away, go to the target
        distance = util.distance(self.ship.loc, self.target.loc) - self.target.radius
        if distance > self.max_dist:
            order = DockingOrder.create_docking_order(self.target, self.ship, self.gamestate, surface_distance=self.max_dist, observer=self)
            self._add_child(order)
            return

        #TODO: multiple goods? transfer from us to them?
        if not self.transfer_effect:
            self.transfer_effect = self._initialize_transfer()
            self.transfer_effect.observe(self)
            self.ship.sector.add_effect(self.transfer_effect)
        # else wait for the transfer effect

    def _initialize_transfer(self) -> effects.TransferCargoEffect:
        assert self.ship.sector is not None
        return effects.TransferCargoEffect.create_transfer_cargo_effect(
                self.resource, self.amount, self.ship, self.target,
                self.ship.sector, self.gamestate,
                transfer_rate=self.transfer_rate())

class TradeCargoToStation(TransferCargo):
    @classmethod
    def create_trade_cargo_to_station[T:"TradeCargoToStation"](cls:Type[T], buyer:core.EconAgent, seller:core.EconAgent, floor_price:float, *args:Any, **kwargs:Any) -> T:
        o = cls.create_transfer_cargo(*args, floor_price, **kwargs)
        o.buyer = buyer
        o.seller = seller
        return o

    def __init__(self, floor_price:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.buyer:core.EconAgent = None # type: ignore
        self.seller:core.EconAgent = None # type: ignore
        self.floor_price = floor_price

    def _initialize_transfer(self) -> effects.TransferCargoEffect:
        assert self.ship.sector is not None
        assert self.buyer == self.gamestate.econ_agents[self.target.entity_id]
        #TODO: what should we do if the buyer doesn't represent the station
        # anymore (might have happened since we started the order)
        return effects.TradeTransferEffect.create_trade_transfer_effect(
                self.buyer, self.seller, econ.buyer_price,
                self.resource, self.amount, self.ship, self.target,
                self.ship.sector, self.gamestate,
                floor_price=self.floor_price,
                transfer_rate=self.transfer_rate())

    def act(self, dt:float) -> None:
        #TODO: what happens if the buyer changes?
        assert self.buyer == self.gamestate.econ_agents.get(self.target.entity_id)
        super().act(dt)

class TradeCargoFromStation(TransferCargo):
    @classmethod
    def create_trade_cargo_from_station[T:"TradeCargoFromStation"](cls:Type[T], buyer:core.EconAgent, seller:core.EconAgent, ceiling_price:float, *args:Any, **kwargs:Any) -> T:
        o = cls.create_transfer_cargo(*args, ceiling_price, **kwargs)
        o.buyer = buyer
        o.seller = seller
        assert(o is not None)
        return o

    def __init__(self, ceiling_price:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.buyer:core.EconAgent = None # type: ignore
        self.seller:core.EconAgent = None # type: ignore
        self.ceiling_price = ceiling_price

    def _initialize_transfer(self) -> effects.TransferCargoEffect:
        assert self.ship.sector is not None
        assert self.seller == self.gamestate.econ_agents[self.target.entity_id]
        #TODO: what should we do if the buyer doesn't represent the station
        # anymore (might have happened since we started the order)
        return effects.TradeTransferEffect.create_trade_transfer_effect(
                self.buyer, self.seller, econ.seller_price,
                self.resource, self.amount, self.target, self.ship,
                self.ship.sector, self.gamestate,
                ceiling_price=self.ceiling_price,
                transfer_rate=self.transfer_rate())

    def act(self, dt:float) -> None:
        #TODO: what happens if the buyer changes?
        assert self.seller == self.gamestate.econ_agents.get(self.target.entity_id)
        super().act(dt)

class DisembarkToEntity(core.OrderObserver, core.Order):
    @staticmethod
    def disembark_to(embark_to:core.SectorEntity, ship:core.Ship, gamestate:core.Gamestate, disembark_dist:float=5e3, disembark_margin:float=5e2) -> DisembarkToEntity:
        if ship.sector is None or ship.sector != embark_to.sector:
            raise ValueError(f'{ship} in {ship.sector} instead of destination {embark_to.sector}')
        hits = ship.sector.spatial_point(ship.loc, max_dist=disembark_dist)
        nearest_dist:np.float64 = np.inf # type: ignore
        nearest = None
        for entity in hits:
            if entity == ship:
                continue
            if np.allclose(entity.velocity, ZERO_VECTOR):
                dist = np.linalg.norm(entity.loc - ship.loc)-entity.radius
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = entity

        return DisembarkToEntity.create_disembark_to_entity(nearest, embark_to, ship, gamestate, disembark_dist=disembark_dist, disembark_margin=disembark_margin)

    @classmethod
    def create_disembark_to_entity[T:"DisembarkToEntity"](cls:Type[T], disembark_from: Optional[core.SectorEntity], embark_to: core.SectorEntity, *args: Any, disembark_dist:float=5e3, disembark_margin:float=5e2, **kwargs: Any) -> T:
        if disembark_from and disembark_from.sector != embark_to.sector:
            raise ValueError(f'from in {disembark_from.sector}, but to is in {embark_to.sector}, they must be colocated')

        o = cls.create_order(*args, disembark_dist=disembark_dist, disembark_margin=disembark_margin, **kwargs)

        o.disembark_from = disembark_from
        if disembark_from is not None:
            o.eow_from = core.EntityOrderWatch(o, disembark_from)
        o.embark_to = embark_to
        o.eow_to = core.EntityOrderWatch(o, embark_to)

        return o

    def __init__(self, *args: Any, disembark_dist:float=5e3, disembark_margin:float=5e2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.disembark_dist = disembark_dist
        self.disembark_margin = disembark_margin

        self.disembark_from:Optional[core.SectorEntity] = None
        self.eow_from:Optional[core.EntityOrderWatch] = None
        self.embark_to:core.SectorEntity = None # type: ignore
        self.eow_to:core.EntityOrderWatch = None # type: ignore

        self.disembark_order:Optional[GoToLocation] = None
        self.embark_order:Optional[GoToLocation] = None

    def _begin(self) -> None:
        # should be upper bound of distance to the disembarkation point
        disembark_loc = self.ship.loc + util.polar_to_cartesian(self.disembark_dist, -self.ship.angle)
        self.init_eta = (
                GoToLocation.compute_eta(self.ship, disembark_loc)
                + GoToLocation.compute_eta(self.ship, self.embark_to.loc)
        )

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def order_complete(self, order:core.Order) -> None:
        assert order == self.embark_order
        self.gamestate.schedule_order_immediate(self)

    def order_cancel(self, order:core.Order) -> None:
        assert order == self.embark_order
        self.gamestate.schedule_order_immediate(self)

    def _is_complete(self) -> bool:
        return self.embark_order is not None and self.embark_order.is_complete()

    def act(self, dt:float) -> None:
        #TODO: should this work across sectors?
        if self.ship.sector is None or self.ship.sector != self.embark_to.sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of destination {self.embark_to.sector}')

        self.embark_order = GoToLocation.goto_entity(self.embark_to, self.ship, self.gamestate, observer=self)
        if self.disembark_from and np.linalg.norm(self.disembark_from.loc - self.ship.loc)-self.disembark_from.radius < self.disembark_dist:
            assert self.disembark_from.sector == self.ship.sector
            # choose a location which is outside disembark_dist
            _, angle = util.cartesian_to_polar(*(self.ship.loc - self.disembark_from.loc))
            target_angle = angle + self.gamestate.random.uniform(-np.pi/2, np.pi/2)
            target_disembark_distance = self.disembark_from.radius+self.disembark_dist+self.disembark_margin
            target_loc = self.disembark_from.loc + util.polar_to_cartesian(target_disembark_distance, target_angle)

            self.disembark_order = GoToLocation.create_go_to_location(
                    target_loc, self.ship, self.gamestate,
                    arrival_distance=self.disembark_margin,
                    min_distance=0.
            )
            self._add_child(self.embark_order, begin=False)
            self._add_child(self.disembark_order, begin=True)
        else:
            self._add_child(self.embark_order, begin=True)

class TravelThroughGate(core.EffectObserver, core.OrderObserver, core.Order):
    # lifecycle has several phases:
    PHASE_TRAVEL_TO_GATE = 1
    PHASE_TRAVEL_OUT_OF_SECTOR = 2
    PHASE_TRAVEL_IN_TO_SECTOR = 3
    PHASE_COMPLETE = 4

    @classmethod
    def create_travel_through_gate[T:"TravelThroughGate"](cls:Type[T], target_gate: sector_entity.TravelGate, *args: Any, position_margin:float=5e2, travel_time:float=5, travel_thrust:float=5e6, max_gate_dist:float=2e3, **kwargs: Any) -> T:
        o = cls.create_order(*args, position_margin=position_margin, travel_time=travel_time, travel_thrust=travel_thrust, max_gate_dist=max_gate_dist, **kwargs)
        o.target_gate = target_gate
        o.eow = core.EntityOrderWatch(o, target_gate)

        return o

    def __init__(self, *args: Any, position_margin:float=5e2, travel_time:float=5, travel_thrust:float=5e6, max_gate_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.target_gate:sector_entity.TravelGate = None # type: ignore
        self.eow:core.EntityOrderWatch = None # type: ignore
        self.position_margin = position_margin
        self.travel_time = travel_time
        self.travel_thrust = travel_thrust
        self.max_gate_dist = max_gate_dist

        self.phase = self.PHASE_TRAVEL_TO_GATE
        self.travel_start_time = 0.
        self.rotate_order:Optional[RotateOrder] = None

        # the velocity we reached when we warped out, saved for when we warp in
        self.warp_velocity = np.array((0.,0.))
        self.warp_out:Optional[effects.WarpOutEffect] = None
        self.warp_in:Optional[effects.WarpInEffect] = None

    def _is_complete(self) -> bool:
        return self.phase == self.PHASE_COMPLETE

    def __str__(self) -> str:
        return f'TravelThroughGate phase {self.phase}'

    def _cancel(self) -> None:
        #TODO: what should happen to the ship?
        if self.warp_out:
            self.warp_out.cancel_effect()
        if self.warp_in:
            self.warp_in.cancel_effect()

    def _begin(self) -> None:
        # get into position and then some time of acceleration "out of system"
        self.init_eta = GoToLocation.compute_eta(self.ship, self.target_gate.loc) + 5

    # core.EffectObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def effect_complete(self, effect:core.Effect) -> None:
        if self.phase == self.PHASE_TRAVEL_OUT_OF_SECTOR:
            assert effect == self.warp_out
        elif self.phase == self.PHASE_TRAVEL_IN_TO_SECTOR:
            assert effect == self.warp_in
        else:
            assert False
        self.gamestate.schedule_order_immediate(self)

    def effect_cancel(self, effect:core.Effect) -> None:
        if self.phase == self.PHASE_TRAVEL_OUT_OF_SECTOR:
            assert effect == self.warp_out
        elif self.phase == self.PHASE_TRAVEL_IN_TO_SECTOR:
            assert effect == self.warp_in
        else:
            assert False
        self.gamestate.schedule_order_immediate(self)

    # core.OrderObserver
    def order_complete(self, order:core.Order) -> None:
        self.gamestate.schedule_order_immediate(self)

    def order_cancel(self, order:core.Order) -> None:
        self.gamestate.schedule_order_immediate(self)


    def _act_travel_to_gate(self, dt:float) -> None:
        """ Handles action during travel to gate phase.

        If we're at the gate, transitions to travelling out of sector,
        otherwise just sets up a goto order to get us into position. """

        if self.ship.sector is None or self.ship.sector != self.target_gate.sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of gate {self.target_gate.sector}')

        if self.rotate_order is not None and self.rotate_order.is_complete():
            # everything is ready for travel, let's goooooo
            self.ship.set_velocity((0., 0.))

            # start the warp out effect where we will be when we're doing gooing
            # s = v_0*t + 1/2 a * t^2
            expected_r =  0.5 * self.travel_thrust * self.travel_time ** 2
            expected_theta = self.target_gate.direction

            expected_loc = self.ship.loc + util.polar_to_cartesian(expected_r, expected_theta)
            self.warp_out = effects.WarpOutEffect.create_warp_out_effect(
                    expected_loc, self.ship.sector, self.gamestate,
                    ttl=self.travel_time,
                    observer=self)
            self.ship.sector.add_effect(self.warp_out)

            self.phase = self.PHASE_TRAVEL_OUT_OF_SECTOR
            self.travel_start_time = self.gamestate.timestamp

            # continue action when the effect completes
            return

        # zero velocity and not eclipsed by the gate
        rel_pos = self.ship.loc - self.target_gate.loc
        rel_r, rel_theta = util.cartesian_to_polar(*rel_pos)
        if util.both_almost_zero(self.ship.velocity) and rel_r < self.max_gate_dist and abs(rel_theta - self.target_gate.direction) < np.pi/2:
            # we're in position point toward the destination
            self.rotate_order = RotateOrder.create_rotate_order(self.target_gate.direction, self.ship, self.gamestate, observer=self)
            self._add_child(self.rotate_order)

            # continue action when the rotation is complete
            return

        # we're not in position, set up a goto order to get us into position
        desired_r = self.gamestate.random.uniform(1e3, self.max_gate_dist-self.position_margin)
        desired_theta = self.target_gate.direction + self.gamestate.random.uniform(-np.pi/2, np.pi/2)
        desired_position = np.array(util.polar_to_cartesian(desired_r, desired_theta)) + self.target_gate.loc
        goto_order = GoToLocation.create_go_to_location(
                desired_position, self.ship, self.gamestate,
                arrival_distance=self.position_margin, min_distance=0.,
                observer=self
        )
        self._add_child(goto_order)
        # continue action when the goto is complete

    def _act_travel_out_of_sector(self, dt:float) -> None:
        """ Handles action while traveling out of sector.

        Mostly just for effect, we accelerate "out of the sector". Once that's
        done enough, we warp the player ship into another sector, swapping out
        physics bodies and such. """

        if self.ship.sector is None or self.ship.sector != self.target_gate.sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of gate {self.target_gate.sector}')
        # go fast! until we're "out of sector"
        if self.gamestate.timestamp - self.travel_start_time > self.travel_time:
            # set position with enough runway to come to a full stop
            min_r = self.target_gate.destination.radius * 2
            max_r = self.target_gate.destination.radius * 2.5
            # 180 the direction so we come in on the opposite side of the
            # destination sector
            min_theta = np.pi + self.target_gate.direction - math.radians(5)
            max_theta = np.pi + self.target_gate.direction + math.radians(5)

            r = self.gamestate.random.uniform(min_r, max_r)
            theta = self.gamestate.random.uniform(min_theta, max_theta)
            loc = util.polar_to_cartesian(r, theta)

            #TODO: should setting these happen in the post_tick phase?
            self.ship.set_loc(loc)
            self.warp_velocity = np.copy(self.ship.velocity)
            self.ship.set_velocity((0., 0.))

            # move from current sector to destination sector
            self.ship.migrate(self.target_gate.destination)

            self.warp_in = effects.WarpInEffect.create_warp_in_effect(
                    np.copy(self.ship.loc), self.ship.sector, self.gamestate,
                    observer=self)
            self.ship.sector.add_effect(self.warp_in)
            self.phase = self.PHASE_TRAVEL_IN_TO_SECTOR

            # continue action once the effect completes
        else:
            # lets gooooooo
            self.ship.apply_force(self.target_gate.direction_vector * self.travel_thrust, True)
            #TODO: we want to continue applying thrust for the entire interval
            next_ts = self.travel_start_time + self.travel_time
            self.gamestate.schedule_order(next_ts, self)

    def _act_travel_in_to_sector(self, dt:float) -> None:
        if self.ship.sector is None or self.ship.sector != self.target_gate.destination:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of gate destination {self.target_gate.destination}')
        if self.gamestate.timestamp - self.travel_start_time > self.travel_time:
            #TODO: theoretically we should get to zero velocity, but this makes
            # a lot of assumptions about our starting velocity and precise
            # application of force over time, so it's possible this won't work

            # should already be stopped, but let's make sure
            self.ship.phys.velocity = (0., 0.)
            self.phase = self.PHASE_COMPLETE

            # schedule another tick to get cleaned up
            self.gamestate.schedule_order_immediate(self)
        else:
            self.ship.apply_force(-1 * self.target_gate.direction_vector * self.travel_thrust, True)
            #TODO: we want to continue applying thrust for the entire interval
            next_ts = self.travel_start_time + self.travel_time
            self.gamestate.schedule_order(next_ts, self)

    def act(self, dt:float) -> None:
        if self.phase == self.PHASE_TRAVEL_TO_GATE:
            self._act_travel_to_gate(dt)
        elif self.phase == self.PHASE_TRAVEL_OUT_OF_SECTOR:
            self._act_travel_out_of_sector(dt)
        elif self.phase == self.PHASE_TRAVEL_IN_TO_SECTOR:
            self._act_travel_in_to_sector(dt)
        else:
            raise ValueError(f'unknown gate travel phase {self.phase}')

class DockingOrder(core.OrderObserver, core.Order):
    """ Dock at an entity.

    That is, go to at a point within some docking distance of the entity.
    """

    @staticmethod
    def compute_eta(ship:core.Ship, target:core.SectorEntity) -> float:
        return GoToLocation.compute_eta(ship, target.loc) + 15

    @classmethod
    def create_docking_order[T:"DockingOrder"](cls:Type[T], target:core.SectorEntity, *args:Any, surface_distance:float=7.5e2, approach_distance:float=1e4, wait_time:float=5., **kwargs:Any) -> T:
        o = cls.create_order(*args, surface_distance=surface_distance, approach_distance=approach_distance, wait_time=wait_time, **kwargs)
        o.target = target
        o.eow = core.EntityOrderWatch(o, target)

        return o

    def __init__(self, *args:Any, surface_distance:float=7.5e2, approach_distance:float=1e4, wait_time:float=5., **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        if approach_distance <= surface_distance:
            raise ValueError(f'{approach_distance=} must be greater than {surface_distance=}')
        self.target:core.SectorEntity = None # type: ignore
        self.eow:core.EntityOrderWatch = None # type: ignore
        self.surface_distance = surface_distance
        self.approach_distance = approach_distance
        self.wait_time = wait_time
        self.next_arrival_attempt_time = 0.
        self.started_waiting = np.inf

        # wrap target and observe it
        # on cancel or complete unobserve it
        # on destroy or migrate cancel us

    def _begin(self) -> None:
        # need to get roughly to the target and then time for final approach
        self._init_eta = DockingOrder.compute_eta(self.ship, self.target)

    def _complete(self) -> None:
        if self.ship.captain is not None:
            self.gamestate.trigger_event(
                [self.ship.captain],
                self.gamestate.event_manager.e(events.Events.DOCKED),
                {
                    self.gamestate.event_manager.ck(events.ContextKeys.TARGET): self.target.short_id_int(),
                    self.gamestate.event_manager.ck(events.ContextKeys.SHIP): self.ship.short_id_int(),
                },
            )

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def order_complete(self, order:core.Order) -> None:
        self.gamestate.schedule_order_immediate(self)

    def order_cancel(self, order:core.Order) -> None:
        self.gamestate.schedule_order_immediate(self)

    def _is_complete(self) -> bool:
        distance_to_target = util.distance(self.ship.loc, self.target.loc)
        return distance_to_target < self.surface_distance + self.target.radius

    def act(self, dt:float) -> None:
        if self.ship.sector != self.target.sector:
            self.cancel_order()

        if self.gamestate.timestamp < self.next_arrival_attempt_time:
            return

        distance_to_target = util.distance(self.ship.loc, self.target.loc)
        if distance_to_target > self.approach_distance + self.target.radius:
            self.logger.debug('embarking to target')
            goto_order = GoToLocation.goto_entity(self.target, self.ship, self.gamestate, surface_distance=self.approach_distance, observer=self)
            self._add_child(goto_order)
            self.started_waiting = np.inf
        else:
            try:
                goto_order = GoToLocation.goto_entity(self.target, self.ship, self.gamestate, surface_distance=self.surface_distance, empty_arrival=True, observer=self)
                self.started_waiting = np.inf
            except GoToLocation.NoEmptyArrivalError:
                self.logger.debug(f'arrival zone is full, waiting. waited {self.gamestate.timestamp - self.started_waiting:.0f}s so far')
                if self.started_waiting > self.gamestate.timestamp:
                    self.started_waiting = self.gamestate.timestamp
                self.next_arrival_attempt_time = self.gamestate.timestamp + self.wait_time
                self.gamestate.schedule_order(self.next_arrival_attempt_time, self)
            else:
                self.logger.debug(f'arrival zone empty, beginning final approach')
                self._add_child(goto_order)

class LocationExploreOrder(core.OrderObserver, core.Order):
    def __init__(self, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], *args:Any, **kwargs:Any) -> None:
        self.sector_id = sector_id
        self.loc = loc

        self.goto_order:Optional[movement.GoToLocation] = None
        self.scan_order:Optional[sensors.SensorScanOrder] = None

    # core.OrderObserver

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def order_cancel(self, order:core.Order) -> None:
        if order == self.goto_order:
            self.goto_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        elif order == self.scan_order:
            self.scan_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        else:
            raise ValueError(f'unexpected order event for {order}')

    def order_complete(self, order:core.Order) -> None:
        if order == self.goto_order:
            self.goto_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        elif order == self.scan_order:
            self.scan_order = None
            # we sucessfully scanned at the, we assume according to act logic
            # below, correct location. thus we are done.
            self.complete_order()
        else:
            raise ValueError(f'unexpected order event for {order}')

    # core.Order

    def _begin(self) -> None:
        self.init_eta = movement.GoToLocation.compute_eta(self.ship, self.loc)

    def act(self, dt:float) -> None:
        #TODO: is this a safe assert?
        assert self.ship.sector
        if self.ship.sector.entity_id != self.sector_id:
            #TODO: get to target sector
            raise ValueError(f'ship must be in the target sector!')
        elif util.distance(self.ship.loc, self.loc) > self.ship.sector.hex_size:
            # go to target location
            assert(self.goto_order is None)
            self.goto_order = GoToLocation.create_go_to_location(self.loc, self.ship, self.gamestate, surface_distance=self.ship.sector.hex_size/2.0)
            self.goto_order.observe(self)
            self._add_child(self.goto_order)
        else:
            # do a sensor scan
            assert(self.scan_order is None)
            self.scan_order = sensors.SensorScanOrder(self.gamestate)
            self.scan_order.observe(self)
            self._add_child(self.scan_order)
