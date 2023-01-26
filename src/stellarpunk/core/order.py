""" Orders and stuff that facilitate very specific activities in a Sector. """

import abc
import logging
import collections
from typing import Any, Optional, List, Tuple, Deque, TYPE_CHECKING

import numpy as np

from stellarpunk import util

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
        self.observers:List[EffectObserver] = []

        self.logger = logging.getLogger(util.fullname(self))

        if observer is not None:
            self.observe(observer)

    def observe(self, observer:EffectObserver) -> None:
        self.observers.append(observer)

    def _begin(self) -> None:
        pass

    def _complete(self) -> None:
        pass

    def _cancel(self) -> None:
        pass

    @abc.abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]:
        """ returns a 4-tuple bounding box ul_x, ul_y, lr_x, lr_y """
        pass

    def is_complete(self) -> bool:
        return True

    def begin_effect(self) -> None:
        self.started_at = self.gamestate.timestamp
        self._begin()

        for observer in self.observers:
            observer.effect_begin(self)

    def complete_effect(self) -> None:
        self.completed_at = self.gamestate.timestamp
        self._complete()

        self.logger.debug(f'effect {self} in {self.sector.short_id()} complete in {self.gamestate.timestamp - self.started_at:.2f}')

        for observer in self.observers:
            observer.effect_complete(self)

        self.sector.remove_effect(self)

    def cancel_effect(self) -> None:
        self.gamestate.unschedule_effect(self)
        try:
            self.sector.remove_effect(self)
        except ValueError:
            # effect might already have been removed from the queue
            pass

        self._cancel()

        for observer in self.observers:
            observer.effect_cancel(self)

    def act(self, dt:float) -> None:
        # by default we'll just complete the effect if it's done
        if self.is_complete():
            self.complete_effect()


class OrderLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, ship:"Ship", *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.ship = ship

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        assert self.ship.sector is not None
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
        self.child_orders:Deque[Order] = collections.deque()

        self.observers:List[OrderObserver] = []
        if observer is not None:
            self.observe(observer)

    def __str__(self) -> str:
        return f'{self.__class__} for {self.ship}'

    def observe(self, observer:OrderObserver) -> None:
        self.observers.append(observer)

    def _add_child(self, order:"Order", begin:bool=True) -> None:
        self.child_orders.appendleft(order)
        self.ship.prepend_order(order, begin=begin)

    def to_history(self) -> dict:
        return {"o": self.o_name}

    def is_complete(self) -> bool:
        """ Indicates that this Order is ready to complete and be removed from
        the order queue. """
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
        self.completed_at = self.gamestate.timestamp
        self._complete()

        for observer in self.observers:
            observer.order_complete(self)

    def cancel_order(self) -> None:
        """ Called when an order is removed from the order queue, but not
        because it's complete. Note the order _might_ be complete in this case.
        """
        self.gamestate.unschedule_order(self)
        for order in self.child_orders:
            order.cancel_order()
            try:
                self.ship.remove_order(order)
            except ValueError:
                # order might already have been removed from the queue
                pass

        try:
            self.ship.remove_order(self)
        except ValueError:
            # order might already have been removed from the queue
            pass

        self._cancel()

        for observer in self.observers:
            observer.order_cancel(self)

    def act(self, dt:float) -> None:
        """ Performs one immediate tick's worth of action for this order """
        pass
