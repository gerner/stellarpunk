""" Ship

A special kind of SectorEntity that has mobility, can be ordered, etc. """

import collections
from typing import TypeAlias, Callable, Any, Deque, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from .sector_entity import SectorEntity, ObjectType, HistoryEntry
from .order import Order, NullOrder
from .character import Asset

if TYPE_CHECKING:
    from .gamestate import Gamestate

class PointDefenseSettings:
    def __init__(self) -> None:
        self.rof:float = 0.
        self.muzzle_velocity:float = 0.
        self.dispersion_angle:float = 0.
        self.projectile_ttl:float = 0.
        self.target_max_age:float = 0.
        self.fire_arc:float = 0.
        self.max_angular_velocity:float = 0.

        self.heading:float = 0.

class Ship(SectorEntity, Asset):
    DefaultOrderSig:TypeAlias = "Callable[[Ship, Gamestate], Order]"

    id_prefix = "SHP"
    object_type = ObjectType.SHIP

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # SI units (newtons and newton-meters)
        # max_base_thrust is the underlying maximum
        # max_thrust is the current set max thrust (which can fall below base)
        # max thrust along heading vector
        self.max_base_thrust = 0.
        self.max_thrust = 0.
        # max thrust in any direction
        self.max_fine_thrust = 0.
        # max torque for turning (in newton-meters)
        self.max_torque = 0.

        self._orders: Deque[Order] = collections.deque()
        self.default_order_fn:Ship.DefaultOrderSig = lambda ship, gamestate: NullOrder.create_null_order(ship, gamestate)


    def _destroy(self) -> None:
        self._clear_orders()

    def get_history(self) -> Sequence[HistoryEntry]:
        return self.history

    def to_history(self, timestamp:float) -> HistoryEntry:
        order_hist = self._orders[0].to_history() if self._orders else None
        return HistoryEntry(
                self.id_prefix,
                self.entity_id, timestamp,
                tuple(self.phys.position), self.radius, self.angle,
                tuple(self.phys.velocity), self.angular_velocity,
                tuple(self.phys.force), self.phys.torque,
                order_hist,
        )

    def max_speed(self) -> float:
        return self.max_base_thrust / self.mass * 30

    def max_acceleration(self) -> float:
        return self.max_thrust / self.mass

    def max_fine_acceleration(self) -> float:
        return self.max_fine_thrust / self.mass

    def max_angular_acceleration(self) -> float:
        return self.max_torque / self.moment

    def apply_force(self, force: Union[Sequence[float], npt.NDArray[np.float64]], persistent:bool) -> None:
        self.phys.force = cymunk.vec2d.Vec2d(*force)
        self.sensor_settings.set_thrust(self.phys.force.length)

    def apply_torque(self, torque: float, persistent:bool) -> None:
        self.phys.torque = torque

    def set_loc(self, loc: Union[Sequence[float], npt.NDArray[np.float64]]) -> None:
        self.phys.position = (loc[0], loc[1])

    def set_velocity(self, velocity: Union[Sequence[float], npt.NDArray[np.float64]]) -> None:
        self.phys.velocity = (velocity[0], velocity[1])

    def set_angle(self, angle: float) -> None:
        self.phys.angle = angle

    def set_angular_velocity(self, angular_velocity:float) -> None:
        self.phys.angular_velocity = angular_velocity

    def default_order(self, gamestate: "Gamestate") -> Order:
        return self.default_order_fn(self, gamestate)

    def prepend_order(self, order:Order, begin:bool=True) -> None:
        co = self.current_order()
        if co is not None:
            #TODO: should we do anything else to suspend the current order?
            if co.gamestate.is_order_scheduled(co):
                co.gamestate.unschedule_order(co)

        self._orders.appendleft(order)
        if begin:
            order.begin_order()

    def append_order(self, order:Order, begin:bool=False) -> None:
        self._orders.append(order)
        if begin:
            order.begin_order()

    def remove_order(self, order:Order) -> None:
        if order in self._orders:
            self._orders.remove(order)

        co = self.current_order()
        if co is not None and not co.gamestate.is_order_scheduled(co):
            co.gamestate.schedule_order_immediate(co)

    def _clear_orders(self) -> None:
        while self._orders:
            self._orders[0].cancel_order()

    def clear_orders(self, gamestate:"Gamestate") -> None:
        self._clear_orders()
        self.prepend_order(self.default_order(gamestate))

    def pop_current_order(self) -> None:
        self._orders.popleft()

    def complete_current_order(self) -> None:
        order = self._orders.popleft()
        order.complete_order()

    def top_order(self) -> Optional[Order]:
        current_order = self.current_order()
        if current_order is None:
            return None
        while current_order.parent_order is not None:
            current_order = current_order.parent_order
        return current_order

    def current_order(self) -> Optional[Order]:
        if len(self._orders) > 0:
            return self._orders[0]
        else:
            return None

class Missile(Ship):
    id_prefix = "MSL"
    object_type = ObjectType.MISSILE

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # missiles don't run transponders
        self.firer:Optional[SectorEntity] = None

