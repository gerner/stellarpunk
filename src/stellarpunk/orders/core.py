""" Orders that can be given to ships. """

from __future__ import annotations

import math
import uuid
import enum
from typing import Optional, Any, Type

import numpy as np
import numpy.typing as npt

from stellarpunk import util, core, effects, econ, events, sensors, intel
from stellarpunk.core import sector_entity
from stellarpunk.narrative import director
from . import movement

from .movement import GoToLocation, RotateOrder
from .steering import ZERO_VECTOR

class MineOrder(core.OrderObserver, core.EffectObserver, core.Order):
    @classmethod
    def create_mine_order[T:"MineOrder"](cls:Type[T], target_intel: intel.AsteroidIntel, amount: float, *args: Any, max_dist:float=2e3, **kwargs: Any) -> MineOrder:
        o = cls.create_order(*args, max_dist=max_dist, **kwargs)
        o.target_intel = target_intel
        o.amount = min(amount, o.ship.cargo_capacity)
        return o

    def __init__(self, *args: Any, max_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_intel:intel.AsteroidIntel = None # type: ignore
        self.target_image:Optional[core.AbstractSensorImage] = None
        self.max_dist = max_dist
        self.amount = 0.0
        self.mining_effect:Optional[effects.MiningEffect] = None
        self.mining_rate = 2e1

    def estimate_eta(self) -> float:
        assert self.ship.sector
        if self.ship.sector.entity_id != self.target_intel.sector_id:
            return NavigateOrder.compute_eta(self.ship, self.target_intel.sector_id) + DockingOrder.compute_eta(self.ship, self.target_intel.loc, starting_loc=ZERO_VECTOR) + 5 + self.amount / self.mining_rate

        if self.target_image is None:
            self.target_image = self.ship.sector.sensor_manager.target_from_identity(self.target_intel.create_sensor_identity(), self.ship, self.target_intel.loc)
        docking_eta = DockingOrder.compute_eta(self.ship, self.target_image.loc)
        if docking_eta > 0:
            return docking_eta + 5 + self.amount / self.mining_rate
        elif self.mining_effect is not None:
            return self.mining_effect.estimate_eta()
        else:
            raise Exception("no docking time, but also no mining effect")

    def is_cancellable(self) -> bool:
        return self.mining_effect is None

    def _begin(self) -> None:
        self.init_eta = self.estimate_eta()

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
        if not self.ship.sector or self.ship.sector.entity_id != self.target_intel.sector_id:
            #TODO: what if we don't have any path to the target?
            self._add_child(NavigateOrder.create_order(self.ship, self.gamestate, self.target_intel.sector_id, observer=self))
            return
        # we know we're in the same sector as the target
        if self.target_image is None:
            self.target_image = self.ship.sector.sensor_manager.target_from_identity(self.target_intel.create_sensor_identity(), self.ship, self.target_intel.loc)
        else:
            self.target_image.update()

        if not self.target_image.is_active():
            # asteroid isn't there anymore!
            self.cancel_order()
            return

        # grab resources from the asteroid and add to our cargo
        distance = util.distance(self.ship.loc, self.target_image.loc) - self.target_image.identity.radius
        if distance > self.max_dist:
            order = DockingOrder.create_docking_order(self.ship, self.gamestate, target_image=self.target_image, surface_distance=self.max_dist, observer=self)
            self._add_child(order)
            return

        # we must be close enough to identify the target
        assert self.target_image.currently_identified

        if movement.KillVelocityOrder.in_motion(self.ship):
            self._add_child(movement.KillVelocityOrder.create_kill_velocity_order(self.ship, self.gamestate, observer=self))
            return

        if not self.mining_effect:
            assert self.ship.phys.torque == 0.
            actual_target = self.gamestate.get_entity(self.target_image.identity.entity_id, sector_entity.Asteroid)
            assert actual_target.sector == self.ship.sector

            self.mining_effect = effects.MiningEffect.create_transfer_cargo_effect(
                    actual_target.resource, self.amount, actual_target, self.ship, self.ship.sector, self.gamestate, transfer_rate=self.mining_rate, observer=self)
            self.ship.sector.add_effect(self.mining_effect)

        # else wait for the mining effect

class TransferCargo(core.OrderObserver, core.EffectObserver, core.Order):
    @classmethod
    def transfer_rate(cls) -> float:
        return 1e2

    @classmethod
    def create_transfer_cargo[T:"TransferCargo"](cls:Type[T], target_intel:intel.SectorEntityIntel, resource: int, amount: float, *args:Any, max_dist:float=2e3, **kwargs:Any) -> T:
        o = cls.create_order(*args, resource, amount, **kwargs)
        o.target_intel = target_intel
        return o

    def __init__(self, resource: int, amount: float, *args: Any, max_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.target_intel:intel.SectorEntityIntel = None # type: ignore
        self.target_image:Optional[core.AbstractSensorImage] = None
        self.resource = resource
        self.amount = amount
        self.transferred = 0.
        self.max_dist = max_dist

        self.transfer_effect:Optional[core.Effect] = None

    def _begin(self) -> None:
        #TODO: multi-sector estimate
        self.init_eta = (
                DockingOrder.compute_eta(self.ship, self.target_intel.loc)
                + self.amount / self.transfer_rate()
        )

    def _cancel(self) -> None:
        if self.transfer_effect:
            self.transfer_effect.cancel_effect()

    def is_cancellable(self) -> bool:
        return self.transfer_effect is None

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
        if self.ship.sector is None or self.ship.sector.entity_id != self.target_intel.sector_id:
            #TODO: what if we don't have any path to the target?
            self._add_child(NavigateOrder.create_order(self.ship, self.gamestate, self.target_intel.sector_id, observer=self))
            return

        if self.target_image is None:
            self.target_image = self.ship.sector.sensor_manager.target_from_identity(self.target_intel.create_sensor_identity(), self.ship, self.target_intel.loc)
        else:
            self.target_image.update()

        if not self.target_image.is_active():
            # target doesn't exist any more
            self.cancel_order()
            return

        # if we're too far away, go to the target
        distance = util.distance(self.ship.loc, self.target_image.loc) - self.target_image.identity.radius
        if distance > self.max_dist:
            order = DockingOrder.create_docking_order(self.ship, self.gamestate, target_image=self.target_image, surface_distance=self.max_dist, observer=self)
            self._add_child(order)
            return

        #TODO: multiple goods? transfer from us to them?
        if not self.transfer_effect:
            transfer_effect = self._initialize_transfer()
            if transfer_effect is None:
                self.cancel_order()
                return
            self.transfer_effect = transfer_effect
            self.transfer_effect.observe(self)
            self.ship.sector.add_effect(self.transfer_effect)
        # else wait for the transfer effect

    def _initialize_transfer(self) -> Optional[effects.TransferCargoEffect]:
        assert self.ship.sector is not None
        assert self.target_image
        assert self.target_image.detected
        assert self.target_image.currently_identified
        assert self.target_image.is_active()
        actual_target = self.gamestate.get_entity(self.target_image.identity.entity_id, core.SectorEntity)
        return effects.TransferCargoEffect.create_transfer_cargo_effect(
                self.resource, self.amount, self.ship, actual_target,
                self.ship.sector, self.gamestate,
                transfer_rate=self.transfer_rate())

class TradeCargoToStation(TransferCargo):
    @classmethod
    def create_trade_cargo_to_station[T:"TradeCargoToStation"](cls:Type[T], buyer_id:uuid.UUID, seller:core.EconAgent, floor_price:float, *args:Any, **kwargs:Any) -> T:
        o = cls.create_transfer_cargo(*args, floor_price, **kwargs)
        o.buyer_id = buyer_id
        o.seller = seller
        return o

    def __init__(self, floor_price:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.buyer_id:uuid.UUID = None # type: ignore
        self.seller:core.EconAgent = None # type: ignore
        self.floor_price = floor_price

    def _initialize_transfer(self) -> Optional[effects.TransferCargoEffect]:
        assert self.target_image
        assert self.target_image.detected
        assert self.target_image.currently_identified
        assert self.target_image.is_active()
        assert self.ship.sector is not None

        # check that the expected econ agent represents this station
        # otherwise bail, some agenda can try again
        actual_agent = self.gamestate.econ_agents[self.target_image.identity.entity_id]
        if actual_agent.entity_id != self.buyer_id:
            return None

        actual_target = self.gamestate.get_entity(self.target_image.identity.entity_id, core.SectorEntity)
        return effects.TradeTransferEffect.create_trade_transfer_effect(
                actual_agent, self.seller, econ.buyer_price,
                self.resource, self.amount, self.ship, actual_target,
                self.ship.sector, self.gamestate,
                floor_price=self.floor_price,
                transfer_rate=self.transfer_rate())

class TradeCargoFromStation(TransferCargo):
    @classmethod
    def create_trade_cargo_from_station[T:"TradeCargoFromStation"](cls:Type[T], buyer:core.EconAgent, seller_id:uuid.UUID, ceiling_price:float, *args:Any, **kwargs:Any) -> T:
        o = cls.create_transfer_cargo(*args, ceiling_price, **kwargs)
        o.buyer = buyer
        o.seller_id = seller_id
        assert(o is not None)
        return o

    def __init__(self, ceiling_price:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.buyer:core.EconAgent = None # type: ignore
        self.seller_id:uuid.UUID = None # type: ignore
        self.ceiling_price = ceiling_price

    def _initialize_transfer(self) -> Optional[effects.TransferCargoEffect]:
        assert self.target_image
        assert self.target_image.detected
        assert self.target_image.currently_identified
        assert self.target_image.is_active()
        assert self.ship.sector is not None

        # check that the econ agent for the station is what we expect
        # otherwise bail and let agenda try again
        actual_agent = self.gamestate.econ_agents[self.target_image.identity.entity_id]
        if actual_agent.entity_id != self.seller_id:
            return None

        actual_target = self.gamestate.get_entity(self.target_image.identity.entity_id, core.SectorEntity)
        return effects.TradeTransferEffect.create_trade_transfer_effect(
                self.buyer, actual_agent, econ.seller_price,
                self.resource, self.amount, actual_target, self.ship,
                self.ship.sector, self.gamestate,
                ceiling_price=self.ceiling_price,
                transfer_rate=self.transfer_rate())


class TravelThroughGate(core.EffectObserver, core.OrderObserver, core.Order):
    # lifecycle has several phases:
    class Phase(enum.IntEnum):
        PHASE_TRAVEL_TO_GATE = enum.auto()
        PHASE_TRAVEL_OUT_OF_SECTOR = enum.auto()
        PHASE_TRAVEL_IN_TO_SECTOR = enum.auto()
        PHASE_COMPLETE = enum.auto()

    @classmethod
    def compute_eta(cls, ship:core.Ship, target_gate:intel.TravelGateIntel, starting_loc:Optional[npt.NDArray[np.float64]]=None) -> float:
        return GoToLocation.compute_eta(ship, target_gate.loc, starting_loc=starting_loc) + 5

    @classmethod
    def create_travel_through_gate[T:"TravelThroughGate"](cls:Type[T], target_gate: intel.TravelGateIntel, *args: Any, position_margin:float=5e2, travel_time:float=5, travel_thrust:float=5e6, max_gate_dist:float=2e3, **kwargs: Any) -> T:
        o = cls.create_order(*args, position_margin=position_margin, travel_time=travel_time, travel_thrust=travel_thrust, max_gate_dist=max_gate_dist, **kwargs)
        o.target_gate = target_gate
        assert o.ship.sector
        o.target_gate_image = o.ship.sector.sensor_manager.target_from_identity(target_gate.create_sensor_identity(), o.ship, target_gate.loc)

        return o

    def __init__(self, *args: Any, position_margin:float=5e2, travel_time:float=5, travel_thrust:float=5e6, max_gate_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.target_gate:intel.TargetGateIntel = None # type: ignore
        self.target_gate_image:Optional[core.AbstractSensorImage] = None
        self.position_margin = position_margin
        self.travel_time = travel_time
        self.travel_thrust = travel_thrust
        self.max_gate_dist = max_gate_dist

        self.phase = self.Phase.PHASE_TRAVEL_TO_GATE
        self.travel_start_time = 0.

        self.warp_out:Optional[effects.WarpOutEffect] = None
        self.warp_in:Optional[effects.WarpInEffect] = None

    def _is_complete(self) -> bool:
        return self.phase == self.Phase.PHASE_COMPLETE

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
        self.init_eta = TravelThroughGate.compute_eta(self.ship, self.target_gate)

    def is_cancellable(self) -> bool:
        return self.warp_out is None and self.warp_in is None

    # core.EffectObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def effect_complete(self, effect:core.Effect) -> None:
        assert effect in (self.warp_out, self.warp_in)
        if effect == self.warp_out:
            self.warp_out = None
            assert self.phase in (self.Phase.PHASE_TRAVEL_OUT_OF_SECTOR, self.Phase.PHASE_TRAVEL_IN_TO_SECTOR)
        elif effect == self.warp_in:
            self.warp_in = None
            assert self.phase in (self.Phase.PHASE_TRAVEL_IN_TO_SECTOR, self.Phase.PHASE_COMPLETE)
        else:
            raise ValueError(f'got unexpected effect {effect}')

        self.gamestate.schedule_order_immediate(self)

    def effect_cancel(self, effect:core.Effect) -> None:
        assert effect in (self.warp_out, self.warp_in)
        if effect == self.warp_out:
            self.warp_out = None
            assert self.phase in (self.Phase.PHASE_TRAVEL_OUT_OF_SECTOR, self.Phase.PHASE_TRAVEL_IN_TO_SECTOR)
        elif effect == self.warp_in:
            self.warp_in = None
            assert self.phase in (self.Phase.PHASE_TRAVEL_IN_TO_SECTOR, self.Phase.PHASE_COMPLETE)
        else:
            raise ValueError(f'got unexpected effect {effect}')

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

        if self.ship.sector is None or self.ship.sector.entity_id != self.target_gate.sector_id:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of gate {self.target_gate.sector_id}')

        rel_pos = self.ship.loc - self.target_gate.loc
        rel_r, rel_theta = util.cartesian_to_polar(*rel_pos)

        if not util.both_almost_zero(self.ship.velocity) or rel_r > self.max_gate_dist or abs(rel_theta - self.target_gate.direction) > np.pi/2:
            # not zero velocity or not near gate or eclipsed by the gate
            # we're not in position, set up a goto order to get us into position
            desired_r = self.gamestate.random.uniform(1e3, self.max_gate_dist-self.position_margin)
            desired_theta = self.target_gate.direction + self.gamestate.random.uniform(-np.pi/2, np.pi/2)
            desired_position = np.array(util.polar_to_cartesian(desired_r, desired_theta)) + self.target_gate.loc
            goto_order = GoToLocation.create_go_to_location(
                    desired_position, self.ship, self.gamestate,
                    arrival_distance=self.position_margin, min_distance=0.,
                    observer=self
            )
            self.logger.info(f'made goto order {goto_order}')
            self._add_child(goto_order)
            # continue action when the goto is complete
            return

        elif not util.isclose(util.normalize_angle(self.ship.angle - self.target_gate.direction), 0.0):
            # we're in position point, but not pointing toward the destination
            rotate_order = RotateOrder.create_rotate_order(self.target_gate.direction, self.ship, self.gamestate, observer=self)
            self._add_child(rotate_order)

            # continue action when the rotation is complete
            return

        else:
            assert self.target_gate_image
            assert self.target_gate_image.detected
            assert self.target_gate_image.identified

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

            self.phase = self.Phase.PHASE_TRAVEL_OUT_OF_SECTOR
            self.travel_start_time = self.gamestate.timestamp

            # schedule immediately so we can start traveling out of the sector
            self.gamestate.schedule_order_immediate(self)
            return

    def _act_travel_out_of_sector(self, dt:float) -> None:
        """ Handles action while traveling out of sector.

        Mostly just for effect, we accelerate "out of the sector". Once that's
        done enough, we warp the player ship into another sector, swapping out
        physics bodies and such. """

        assert self.target_gate_image
        assert self.target_gate_image.detected
        assert self.target_gate_image.identified

        if self.ship.sector is None or self.ship.sector.entity_id != self.target_gate.sector_id:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of gate {self.target_gate.sector}')

        # we should double check that the gate is still there (sanity check, it
        # is detected, see above)
        actual_gate = self.gamestate.get_entity(self.target_gate.intel_entity_id, sector_entity.TravelGate)

        # go fast! until we're "out of sector"
        if self.gamestate.timestamp - self.travel_start_time > self.travel_time:

            # figure out how far out from the gate we should enter the sector
            # so we end up near it
            # this is how far we expect to travel while slowing down, plus some
            # safety margin so we don't hit the gate
            # s = u * t + 1/2 * a * t
            # f = m * a
            # a = f / m
            expected_travel_distance = util.magnitude(*self.ship.velocity) * self.travel_time + 0.5 * (-self.travel_thrust / self.ship.mass) * self.travel_time * self.travel_time
            enter_loc_offset = util.polar_to_cartesian(expected_travel_distance, np.pi + actual_gate.direction)

            # then add some noise around where the destination gate should be
            # plus the stopping distance, and a safety margin just in case
            min_r = actual_gate.destination.radius*2 + expected_travel_distance + 2e3
            max_r = actual_gate.destination.radius*2.5 + expected_travel_distance + 2e3
            # 180 the direction so we come in on the opposite side of the
            # destination sector
            min_theta = np.pi + actual_gate.direction# - math.radians(5)
            max_theta = np.pi + actual_gate.direction# + math.radians(5)

            r = self.gamestate.random.uniform(min_r, max_r)
            theta = self.gamestate.random.uniform(min_theta, max_theta)
            loc = util.polar_to_cartesian(r, theta)

            #TODO: should setting these happen in the post_tick phase?
            self.ship.set_loc(loc)
            # keep existing velocity, we'll slow down on the other side

            # move from current sector to destination sector
            # make sure to nuke the image, only makes sense in current sector
            self.target_gate_image = None
            self.ship.migrate(actual_gate.destination)

            #TODO: should this go in migrate or sector?
            crew = core.crew(self.ship)
            if crew:
                self.gamestate.trigger_event(
                    crew,
                    self.gamestate.event_manager.e(events.Events.ENTER_SECTOR),
                    {
                        self.gamestate.event_manager.ck(events.ContextKeys.TARGET): actual_gate.destination.short_id_int(),
                        self.gamestate.event_manager.ck(events.ContextKeys.SHIP): self.ship.short_id_int(),
                    },
                )


            # we won't observe this one, it's just a visual effect at this
            # point since we're alrady migrated to the destination sector
            self.warp_in = effects.WarpInEffect.create_warp_in_effect(
                    np.copy(self.ship.loc), self.ship.sector, self.gamestate)
            self.ship.sector.add_effect(self.warp_in)
            self.phase = self.Phase.PHASE_TRAVEL_IN_TO_SECTOR
            self.travel_start_time = self.gamestate.timestamp

            # rotate to face opposite direction of travel and decelerate
            rotate_order = RotateOrder.create_rotate_order(actual_gate.direction+np.pi, self.ship, self.gamestate, observer=self)
            self._add_child(rotate_order)
            self.ship.apply_force(util.polar_to_cartesian(self.travel_thrust, np.pi+actual_gate.direction), True)
            # we'll act again when the rotate order is finished
            return
        else:
            # lets gooooooo
            self.ship.apply_force(util.polar_to_cartesian(self.travel_thrust, actual_gate.direction), True)
            #TODO: we want to continue applying thrust for the entire interval
            next_ts = self.travel_start_time + self.travel_time
            self.gamestate.schedule_order(next_ts, self)

    def _act_travel_in_to_sector(self, dt:float) -> None:
        if self.ship.sector is None or self.ship.sector.entity_id != self.target_gate.destination_id:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of gate destination {self.target_gate.destination_id}')
        if self.gamestate.timestamp - self.travel_start_time > self.travel_time:
            #TODO: theoretically we should get to zero velocity, but this makes
            # a lot of assumptions about our starting velocity and precise
            # application of force over time, so it's possible this won't work

            # should already be stopped, but let's make sure
            self.ship.apply_force((0., 0.), True)
            self.ship.phys.velocity = (0., 0.)
            self.phase = self.Phase.PHASE_COMPLETE

            # schedule another tick to get cleaned up
            self.gamestate.schedule_order_immediate(self)
        else:
            self.ship.apply_force(util.polar_to_cartesian(self.travel_thrust, self.target_gate.direction+np.pi), True)
            next_ts = self.travel_start_time + self.travel_time
            self.gamestate.schedule_order(next_ts, self)

    def act(self, dt:float) -> None:
        if self.target_gate_image:
            self.target_gate_image.update()
            if not self.target_gate_image.is_active():
                # this would be very odd, right? we don't allow destroying travel gates
                #self.cancel_order()
                assert self.target_gate_image.is_active()
                return
        else:
            assert self.ship.sector and self.ship.sector.entity_id == self.target_gate.destination_id

        if self.phase == self.Phase.PHASE_TRAVEL_TO_GATE:
            self._act_travel_to_gate(dt)
        elif self.phase == self.Phase.PHASE_TRAVEL_OUT_OF_SECTOR:
            self._act_travel_out_of_sector(dt)
        elif self.phase == self.Phase.PHASE_TRAVEL_IN_TO_SECTOR:
            self._act_travel_in_to_sector(dt)
        else:
            raise ValueError(f'unknown gate travel phase {self.phase}')

class DockingOrder(core.OrderObserver, core.Order):
    """ Dock at an entity.

    That is, go to at a point within some docking distance of the entity.
    """

    @staticmethod
    def compute_eta(ship:core.Ship, loc:npt.NDArray[np.float64], starting_loc:Optional[npt.NDArray[np.float64]]=None) -> float:
        return GoToLocation.compute_eta(ship, loc, starting_loc=starting_loc) + 15

    @classmethod
    def create_docking_order[T:"DockingOrder"](cls:Type[T], *args:Any, target_id:Optional[uuid.UUID]=None, target_image:Optional[core.AbstractSensorImage]=None, surface_distance:float=7.5e2, approach_distance:float=1e4, wait_time:float=5., **kwargs:Any) -> T:
        if target_image is None and target_id is None:
            raise ValueError("one of target_image or target_id must be specified")
        elif target_image is not None:
            target_id = target_image.identity.entity_id

        o = cls.create_order(*args, target_id=target_id, surface_distance=surface_distance, approach_distance=approach_distance, wait_time=wait_time, **kwargs)
        if target_image is not None:
            o.target_image = target_image

        return o

    def __init__(self, *args:Any, surface_distance:float=7.5e2, approach_distance:float=1e4, wait_time:float=5., target_id:uuid.UUID, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        if approach_distance <= surface_distance:
            raise ValueError(f'{approach_distance=} must be greater than {surface_distance=}')
        self.target_id:uuid.UUID = target_id
        self.target_image:Optional[core.AbstractSensorImage] = None
        self.surface_distance = surface_distance
        self.approach_distance = approach_distance
        self.wait_time = wait_time
        self.next_arrival_attempt_time = 0.
        self.started_waiting = np.inf

        # wrap target and observe it
        # on cancel or complete unobserve it
        # on destroy or migrate cancel us

    def _ensure_target(self) -> core.AbstractSensorImage:
        if self.target_image is not None:
            return self.target_image
        if self.ship.captain is None:
            raise ValueError(f'{self.ship} cannot create a sensor image for {self.target_id} without a captain')
        if self.ship.sector is None:
            raise ValueError(f'{self.ship} cannot create a sensor image for {self.target_id} outside of any sector')
        target_intel = self.ship.captain.intel_manager.get_intel(intel.EntityIntelMatchCriteria(self.target_id), intel.SectorEntityIntel)
        if target_intel is None:
            raise ValueError(f'{self.ship.captain} has no intel for {self.target_id}')
        if target_intel.sector_id != self.ship.sector.entity_id:
            raise ValueError(f'{self.ship} is in {self.ship.sector}, but target is in {target_intel.sector_id}')
        self.target_image = self.ship.sector.sensor_manager.target_from_identity(target_intel.create_sensor_identity(), self.ship, target_intel.loc)
        return self.target_image

    def _begin(self) -> None:
        self.target_image = self._ensure_target()
        # need to get roughly to the target and then time for final approach
        self._init_eta = DockingOrder.compute_eta(self.ship, self.target_image.loc)

    def _complete(self) -> None:
        assert self.target_image
        crew = core.crew(self.ship)
        if crew:
            self.gamestate.trigger_event(
                crew,
                self.gamestate.event_manager.e(events.Events.DOCKED),
                {
                    self.gamestate.event_manager.ck(events.ContextKeys.TARGET): self.target_image.identity.short_id_int(),
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
        if self.target_image is None:
            return False
        if not self.target_image.is_active():
            return False
        distance_to_target = util.distance(self.ship.loc, self.target_image.loc)
        return distance_to_target < self.surface_distance + self.target_image.identity.radius

    def act(self, dt:float) -> None:
        assert self.target_image
        self.target_image.update()
        if not self.target_image.is_active():
            self.cancel_order()

        if self.ship.sector is None or self.ship.sector.entity_id != self.target_image.identity.sector_id:
            self.cancel_order()

        if self.gamestate.timestamp < self.next_arrival_attempt_time:
            return

        distance_to_target = util.distance(self.ship.loc, self.target_image.loc)
        if distance_to_target > self.approach_distance + self.target_image.identity.radius:
            self.logger.debug('embarking to target')
            goto_order = GoToLocation.goto_entity(self.target_image, self.ship, self.gamestate, surface_distance=self.approach_distance, observer=self)
            self._add_child(goto_order)
            self.started_waiting = np.inf
        else:
            try:
                goto_order = GoToLocation.goto_entity(self.target_image, self.ship, self.gamestate, surface_distance=self.surface_distance, empty_arrival=True, observer=self)
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
    @classmethod
    def compute_eta(cls, ship:core.Ship, sector_id:uuid.UUID, loc:npt.NDArray[np.float64]) -> float:
        assert ship.sector
        if ship.sector.entity_id != sector_id:
            nav_eta = NavigateOrder.compute_eta(ship, sector_id)
            # assume we have to cross from the opposite side of the sector
            target_r, target_theta = util.cartesian_to_polar(*loc)
            starting_loc = util.polar_to_cartesian(ship.sector.radius*2.5, target_theta+np.pi)
            goto_eta = movement.GoToLocation.compute_eta(ship, loc, starting_loc=starting_loc)
            return nav_eta + goto_eta
        else:
            return movement.GoToLocation.compute_eta(ship, loc) + 5.0

    def __init__(self, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.loc = loc

        self.navigate_order:Optional[NavigateOrder] = None
        self.goto_order:Optional[movement.GoToLocation] = None
        self.scan_order:Optional[sensors.SensorScanOrder] = None

    # core.OrderObserver

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def order_cancel(self, order:core.Order) -> None:
        if order == self.navigate_order:
            self.navigate_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        elif order == self.goto_order:
            self.goto_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        elif order == self.scan_order:
            self.scan_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        else:
            raise ValueError(f'unexpected order event for {order}')

    def order_complete(self, order:core.Order) -> None:
        if order == self.navigate_order:
            self.navigate_order = None
            self.gamestate.schedule_order_immediate(self, jitter=1.0)
        elif order == self.goto_order:
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
        assert self.ship.sector
        self.init_eta = LocationExploreOrder.compute_eta(self.ship, self.sector_id, self.loc)

    def act(self, dt:float) -> None:
        #TODO: is this a safe assert?
        assert self.ship.sector
        if self.ship.sector.entity_id != self.sector_id:
            assert self.navigate_order is None
            # get to target sector
            self.navigate_order = NavigateOrder.create_order(self.ship, self.gamestate, self.sector_id)
            self.navigate_order.observe(self)
            self._add_child(self.navigate_order)
        elif util.distance(self.ship.loc, self.loc) > self.ship.sector.hex_size:
            # go to target location
            assert(self.goto_order is None)
            self.goto_order = GoToLocation.create_go_to_location(self.loc, self.ship, self.gamestate, arrival_distance=self.ship.sector.hex_size/2.0)
            self.goto_order.observe(self)
            self._add_child(self.goto_order)
        else:
            # do a sensor scan
            assert(self.scan_order is None)
            self.scan_order = sensors.SensorScanOrder.create_order(self.ship, self.gamestate)
            self.scan_order.observe(self)
            self._add_child(self.scan_order)

class NavigateOrder(core.OrderObserver, core.Order):
    @classmethod
    def compute_eta(cls, ship:core.Ship, target_id:uuid.UUID, start_id:Optional[uuid.UUID]=None, starting_loc:Optional[npt.NDArray[np.float64]]=None) -> float:
        if ship.captain is None:
            raise ValueError(f'cannot estimate eta for captainless ship')

        if start_id is None:
            if ship.sector is None:
                raise ValueError(f'no starting sector id and ship is not in a sector')
            start_id = ship.sector.entity_id

        if start_id == target_id:
            return 0.0

        universe_view = intel.UniverseView.create(ship.captain)
        jump_path = universe_view.compute_path(start_id, target_id)

        if jump_path is None:
            return np.inf

        assert len(jump_path) > 0
        out_gate = jump_path[0][1]
        assert out_gate.sector_id == start_id

        time = TravelThroughGate.compute_eta(ship, out_gate, starting_loc=starting_loc)
        last_out_gate = jump_path[0][1]
        for sector, out_gate, _, _ in jump_path[1:]:
            starting_loc = util.polar_to_cartesian(sector.radius*2.25, last_out_gate.direction+np.pi)
            time += TravelThroughGate.compute_eta(ship, out_gate, starting_loc=starting_loc)
            last_out_gate = out_gate
        assert last_out_gate.destination_id == target_id

        return time

    def __init__(self, sector_id:uuid.UUID, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.gate_order:Optional[TravelThroughGate] = None

    @property
    def observer_id(self) -> uuid.UUID:
        return self.order_id

    def order_cancel(self, order:core.Order) -> None:
        if order == self.gate_order:
            self.gate_order = None
            self.gamestate.schedule_order_immediate(self)
        else:
            raise ValueError(f'unexpected order event for {order}')

    def order_complete(self, order:core.Order) -> None:
        if order == self.gate_order:
            self.gate_order = None
            self.gamestate.schedule_order_immediate(self)
        else:
            raise ValueError(f'unexpected order event for {order}')

    # core.Order

    def _is_complete(self) -> bool:
        return self.ship.sector is not None and self.ship.sector.entity_id == self.sector_id

    def _begin(self) -> None:
        self.init_eta = NavigateOrder.compute_eta(self.ship, self.sector_id)

    def act(self, dt:float) -> None:
        assert self.ship.sector
        assert self.ship.sector.entity_id != self.sector_id
        assert self.ship.captain
        # find a path to the target sector
        universe_view = intel.UniverseView.create(self.ship.captain)
        jump_path = universe_view.compute_path(self.ship.sector.entity_id, self.sector_id)
        if jump_path is None:
            self.logger.debug(f'{self.ship.captain} knows of no path from {self.ship.sector} to {self.sector_id}')
            self.cancel_order()
            return
        assert len(jump_path) > 0
        assert(self.gate_order is None)
        self.gate_order = TravelThroughGate.create_travel_through_gate(jump_path[0][1], self.ship, self.gamestate)
        self.gate_order.observe(self)
        self._add_child(self.gate_order)
