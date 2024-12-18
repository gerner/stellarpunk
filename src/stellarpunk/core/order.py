""" Orders and stuff that facilitate very specific activities in a Sector. """

import abc
import logging
import collections
import weakref
from typing import Any, Optional, Tuple, Deque, TYPE_CHECKING, Set

import numpy as np

from stellarpunk import util, core

if TYPE_CHECKING:
    from .sector import Sector
    from .ship import Ship
    from .gamestate import Gamestate


class EffectObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def effect_begin(self, effect:"Effect") -> None:
        pass

    def effect_complete(self, effect:"Effect") -> None:
        pass

    def effect_cancel(self, effect:"Effect") -> None:
        pass


class Effect(abc.ABC):
    def __init__(self, sector:"Sector", gamestate:"Gamestate", observer:Optional[EffectObserver]=None) -> None:
        self.sector = sector
        self.gamestate = gamestate
        self.started_at = -1.
        self.completed_at = -1.
        self.observers:weakref.WeakSet[EffectObserver] = weakref.WeakSet()

        self.logger = logging.getLogger(util.fullname(self))

        if observer is not None:
            self.observe(observer)

    def observe(self, observer:EffectObserver) -> None:
        self.observers.add(observer)

    def unobserve(self, observer:EffectObserver) -> None:
        try:
            self.observers.remove(observer)
        except KeyError:
            pass

    def _begin(self) -> None:
        """ Called when the effect starts.

        This is a great place to schedule the effect for processing.
        """
        pass

    def _complete(self) -> None:
        """ Called when the effect is done and cleaning up. """
        pass

    def _cancel(self) -> None:
        """ Called when the effect is cancelled and cleaning up. """
        pass

    @abc.abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]:
        """ returns a 4-tuple bounding box ul_x, ul_y, lr_x, lr_y """
        pass

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self._is_complete()

    def _is_complete(self) -> bool:
        return False

    def begin_effect(self) -> None:
        """ Triggers the event to start working.

        Called by the sector when the effect is added to it
        """

        self.started_at = self.gamestate.timestamp
        self._begin()

        for observer in self.observers:
            observer.effect_begin(self)

    def complete_effect(self) -> None:
        """ Triggers the event to complete and get cleaned up.

        We assume the effect is not scheduled at this point.

        The effect (or whoever owns the effect) must call this when the
        effect is done. The default act will call this if is_complete.
        """

        if self.completed_at > 0:
            return
        self.completed_at = self.gamestate.timestamp
        self._complete()

        self.logger.debug(f'effect {self} in {self.sector.short_id()} complete in {self.gamestate.timestamp - self.started_at:.2f}')

        for observer in self.observers:
            observer.effect_complete(self)
        self.observers.clear()

        self.sector.remove_effect(self)

    def cancel_effect(self) -> None:
        """ Triggers the event to cancel and get cleaned up.

        The effect might be scheduled at this point.

        Anyone can call this.
        """

        if self.completed_at > 0:
            return
        self.completed_at = self.gamestate.timestamp

        self.gamestate.unschedule_effect(self)
        try:
            self.sector.remove_effect(self)
        except ValueError:
            # effect might already have been removed from the queue
            pass

        self._cancel()

        for observer in self.observers:
            observer.effect_cancel(self)
        self.observers.clear()

    def act(self, dt:float) -> None:
        # by default we'll just complete the effect if it's done
        if self.is_complete():
            self.complete_effect()


class OrderLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, ship:"Ship", *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.ship = weakref.proxy(ship)

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        if self.ship.sector is None:
            return f'{self.ship.short_id()}@None {msg}', kwargs
        else:
            return f'{self.ship.short_id()}@{self.ship.sector.short_id()} {msg}', kwargs


class OrderObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def order_begin(self, order:"Order") -> None:
        pass

    def order_complete(self, order:"Order") -> None:
        pass

    def order_cancel(self, order:"Order") -> None:
        pass


class Order:
    def __init__(self, ship: "Ship", gamestate: "Gamestate", observer:Optional[OrderObserver]=None) -> None:
        self.gamestate = gamestate
        self.ship = ship
        self.logger = OrderLoggerAdapter(
                ship,
                logging.getLogger(util.fullname(self)),
        )
        self.o_name = util.fullname(self)
        self.started_at = -1.
        self.completed_at = -1.
        self.init_eta = np.inf
        self.parent_order:Optional[Order] = None
        self.child_orders:Deque[Order] = collections.deque()

        self.observers:weakref.WeakSet[OrderObserver] = weakref.WeakSet()
        if observer is not None:
            self.observe(observer)

    def __str__(self) -> str:
        return f'{self.__class__} for {self.ship}'

    def estimate_eta(self) -> float:
        if self.started_at > 0:
            return self.init_eta - (self.gamestate.timestamp - self.started_at)
        else:
            return self.init_eta

    def observe(self, observer:OrderObserver) -> None:
        self.observers.add(observer)

    def unobserve(self, observer:OrderObserver) -> None:
        try:
            self.observers.remove(observer)
        except KeyError:
            pass

    def _add_child(self, order:"Order", begin:bool=True) -> None:
        order.parent_order = self
        self.child_orders.appendleft(order)
        self.ship.prepend_order(order, begin=begin)

    def to_history(self) -> dict:
        return {"o": self.o_name}

    def is_complete(self) -> bool:
        """ Indicates that this Order is ready to complete and be removed from
        the order queue. """
        return self.completed_at > 0 or self._is_complete()

    def _is_complete(self) -> bool:
        return False

    def _begin(self) -> None:
        pass

    def _complete(self) -> None:
        pass

    def _cancel(self) -> None:
        pass

    def begin_order(self) -> None:
        """ Called when an order is ready to start acting, at the front of the
        order queue, before the first call to act. This is a good time to
        compute the eta for the order."""
        self.started_at = self.gamestate.timestamp
        self._begin()

        for observer in self.observers:
            observer.order_begin(self)

        self.gamestate.schedule_order_immediate(self)

    def complete_order(self) -> None:
        """ Called when an order is_complete and about to be removed from the
        order queue. """
        if self.completed_at > 0:
            return
        self.completed_at = self.gamestate.timestamp

        try:
            self.ship.remove_order(self)
        except ValueError:
            # order might already have been removed from the queue
            pass

        for order in self.child_orders:
            order.cancel_order()
            try:
                self.ship.remove_order(order)
            except ValueError:
                # order might already have been removed from the queue
                pass
        self.child_orders.clear()

        if self.parent_order:
            self.parent_order.child_orders.remove(self)

        self._complete()
        for observer in self.observers:
            observer.order_complete(self)
        self.observers.clear()
        self.gamestate.unschedule_order(self)

    def cancel_order(self) -> None:
        """ Called when an order is removed from the order queue, but not
        because it's complete. Note the order _might_ be complete in this case.
        """
        if self.completed_at > 0:
            return
        self.completed_at = self.gamestate.timestamp

        try:
            self.ship.remove_order(self)
        except ValueError:
            # order might already have been removed from the queue
            pass

        for order in self.child_orders:
            order.cancel_order()
            try:
                self.ship.remove_order(order)
            except ValueError:
                # order might already have been removed from the queue
                pass
        self.child_orders.clear()

        if self.parent_order:
            self.parent_order.child_orders.remove(self)

        self._cancel()
        for observer in self.observers:
            observer.order_cancel(self)
        self.observers.clear()
        self.gamestate.unschedule_order(self)

    def act(self, dt:float) -> None:
        """ Performs one immediate tick's worth of action for this order """
        pass


