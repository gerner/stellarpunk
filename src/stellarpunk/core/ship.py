""" Ship

A special kind of SectorEntity that has mobility, can be ordered, etc. """

import collections
from typing import TypeAlias, Callable, Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from . import base, sector, character

class Ship(character.Asset, character.CrewedSectorEntity):
    id_prefix = "SHP"

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

        self._orders: collections.deque[base.AbstractOrder] = collections.deque()

    def _destroy(self) -> None:
        self._clear_orders()

    def get_history(self) -> Sequence[sector.HistoryEntry]:
        return self.history

    def set_history(self, hist:Sequence[sector.HistoryEntry]) -> None:
        self.history.clear()
        for h in hist:
            self.history.append(h)

    def to_history(self, timestamp:float) -> sector.HistoryEntry:
        order_hist = self._orders[0].to_history() if self._orders else None
        return sector.HistoryEntry(
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

    #def default_order(self) -> base.AbstractOrder:
    #    return self.default_order_fn(self)

    def prepend_order(self, order:base.AbstractOrder, begin:bool=True) -> None:
        co = self.current_order()
        if co is not None:
            co.pause()

        order.register()
        self._orders.appendleft(order)

        if begin:
            order.begin_order()

    def append_order(self, order:base.AbstractOrder, begin:bool=False) -> None:
        order.register()
        self._orders.append(order)
        if begin:
            order.begin_order()

    def remove_order(self, order:base.AbstractOrder) -> None:
        if order in self._orders:
            self._orders.remove(order)
            order.unregister()

        co = self.current_order()
        if co:
            co.resume()

    def _clear_orders(self) -> None:
        while self._orders:
            self._orders[0].cancel_order()

    def clear_orders(self) -> None:
        self._clear_orders()
        #self.prepend_order(self.default_order())

    def top_order(self) -> Optional[base.AbstractOrder]:
        current_order = self.current_order()
        if current_order is None:
            return None
        while current_order.parent_order is not None:
            current_order = current_order.parent_order
        return current_order

    def current_order(self) -> Optional[base.AbstractOrder]:
        if len(self._orders) > 0:
            return self._orders[0]
        else:
            return None

