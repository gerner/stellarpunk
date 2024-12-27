""" Orders and stuff that facilitate very specific activities in a Sector. """

import abc
import logging
import collections
import uuid
import weakref
from collections.abc import Iterable
from typing import Any, Optional, Tuple, Deque, Set, Type

import numpy as np

from stellarpunk import util
from . import base
from .gamestate import Gamestate
from .sector import Sector
from .ship import Ship

class EffectObserver(base.Observer):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def effect_begin(self, effect:"Effect") -> None:
        pass

    def effect_complete(self, effect:"Effect") -> None:
        pass

    def effect_cancel(self, effect:"Effect") -> None:
        pass


class Effect(base.AbstractEffect, base.Observable):
    @classmethod
    def create_effect[T:"Effect"](cls:Type[T], sector:"Sector", gamestate:Gamestate, *args:Any, **kwargs:Any) -> T:
        effect = cls(*args, gamestate, _check_flag=True, **kwargs)
        effect.sector = sector
        return effect

    def __init__(self, gamestate:"Gamestate", *args:Any, observer:Optional[EffectObserver]=None, _check_flag:bool=False, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        assert(_check_flag)
        self.sector:Sector = None # type: ignore
        self.gamestate = gamestate
        self.started_at = -1.
        self.completed_at = -1.
        self._observers:weakref.WeakSet[EffectObserver] = weakref.WeakSet()

        self.logger = logging.getLogger(util.fullname(self))

        if observer is not None:
            self.observe(observer)

    def register(self) -> None:
        self.gamestate.register_effect(self)

    def unregister(self) -> None:
        self.gamestate.unregister_effect(self)

    @property
    def observers(self) -> Iterable[base.Observer]:
        return self._observers

    def observe(self, observer:EffectObserver) -> None:
        self._observers.add(observer)

    def unobserve(self, observer:EffectObserver) -> None:
        try:
            self._observers.remove(observer)
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

        for observer in self._observers:
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

        for observer in self._observers:
            observer.effect_complete(self)
        self._observers.clear()

        #TODO: do we need to wrap this is a try/catch the way we do with orders?
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

        for observer in self._observers:
            observer.effect_cancel(self)
        self._observers.clear()

    def act(self, dt:float) -> None:
        # by default we'll just complete the effect if it's done
        if self.is_complete():
            self.complete_effect()

    def sanity_check(self, effect_id:uuid.UUID) -> None:
        assert(effect_id == self.effect_id)
        assert(self in self.sector._effects)
        assert(self.sector.entity_id in self.gamestate.entities)

class OrderLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, ship:"Ship", *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.ship = weakref.proxy(ship)

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        if self.ship.sector is None:
            return f'{self.ship.short_id()}@None {msg}', kwargs
        else:
            return f'{self.ship.short_id()}@{self.ship.sector.short_id()} {msg}', kwargs


class OrderObserver(base.Observer):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def order_begin(self, order:"Order") -> None:
        pass

    def order_complete(self, order:"Order") -> None:
        pass

    def order_cancel(self, order:"Order") -> None:
        pass


class Order(base.AbstractOrder, base.Observable):
    @classmethod
    def create_order[T: "Order"](cls:Type[T], ship:"Ship", gamestate:"Gamestate", *args:Any, **kwargs:Any) -> T:
        """ creates an order and initializes it for use.

        we're playing generic type shenanigans to make this available with good
        type hints on all subclasses. notice also how we're "rotating" the
        arguments so we preserve the right order for each __init__ call,
        assuming they forward *args and **kwargs.

        this classmethod create pattern can be matched by subclasses in order
        to separate the creation of a subclass Order object from the
        initialization of that object using arguments we're not allowed to put
        in the constructor (e.g. Entity, other orders, etc.)

        we do ths to facilitate saving and loading orders without needing to
        have the corresponding entity objects loaded at the time we're loading
        this order. """

        # we need to forward *args and **kwargs here because cls likely is not
        # Order and so *args and **kwargs are likely not None
        o = cls(*args, gamestate, _check_flag=True, **kwargs)
        o.initialize_order(ship)
        return o

    def __init__(self, gamestate: "Gamestate", *args:Any, observer:Optional[OrderObserver]=None, _check_flag:bool=False, **kwargs:Any) -> None:
        # we need *args and **kwargs even though they (should be) None because
        # mypy gets confused when we forward *args and **kwargs from
        # create_order in the case that cls is Order

        assert(_check_flag)

        super().__init__(*args, **kwargs)

        self.gamestate = gamestate
        self.ship:"Ship" = None # type: ignore
        self.logger:OrderLoggerAdapter = None # type: ignore
        self.o_name = util.fullname(self)
        self.started_at = -1.
        self.completed_at = -1.
        self.init_eta = np.inf

        self._observers:weakref.WeakSet[OrderObserver] = weakref.WeakSet()
        if observer is not None:
            self.observe(observer)

    def initialize_order(self, ship:"Ship") -> None:
        self.ship = ship
        self.logger = OrderLoggerAdapter(
                ship,
                logging.getLogger(util.fullname(self)),
        )

    def __str__(self) -> str:
        return f'{self.__class__} for {self.ship}'

    def estimate_eta(self) -> float:
        if self.started_at > 0:
            return self.init_eta - (self.gamestate.timestamp - self.started_at)
        else:
            return self.init_eta

    @property
    def observers(self) -> Iterable[base.Observer]:
        return self._observers

    def observe(self, observer:OrderObserver) -> None:
        self._observers.add(observer)

    def unobserve(self, observer:OrderObserver) -> None:
        try:
            self._observers.remove(observer)
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

        for observer in self._observers:
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
        for observer in self._observers:
            observer.order_complete(self)
        self._observers.clear()
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
        for observer in self._observers:
            observer.order_cancel(self)
        self._observers.clear()
        self.gamestate.unschedule_order(self)

    def base_act(self, dt:float) -> None:
        if self == self.ship.current_order():
            if self.is_complete():
                ship = self.ship
                self.logger.debug(f'ship {self.ship.entity_id} completed {self} in {self.gamestate.timestamp - self.started_at:.2f} est {self.init_eta:.2f}')
                self.complete_order()

                next_order = self.ship.current_order()
                if not next_order:
                    self.ship.clear_orders()
                elif not self.gamestate.is_order_scheduled(next_order):
                    #TODO: this is kind of janky, can't we just demand that orders schedule themselves?
                    # what about the order queue being simply a queue?
                    self.gamestate.schedule_order_immediate(next_order)
            else:
                self.act(dt)
        else:
            # else order isn't the front item, so we'll ignore this action
            self.logger.warning(f'got non-front order scheduled action: {self} vs {self.ship.current_order()=}')
            #self.gamestate.counters[core.Counters.NON_FRONT_ORDER_ACTION] += 1

    def sanity_check(self, order_id:uuid.UUID) -> None:
        assert(order_id == self.order_id)
        assert(self in self.ship._orders)
        assert(self.ship.entity_id in self.gamestate.entities)
        if self.parent_order:
            assert(isinstance(self.parent_order, Order))
            assert(self.parent_order.ship == self.ship)
            assert(self.parent_order.order_id in self.gamestate.orders)
        for child_order in self.child_orders:
            assert(isinstance(child_order, Order))
            assert(child_order.ship == self.ship)
            assert(child_order.order_id in self.gamestate.orders)

    def pause(self) -> None:
        if self.gamestate.is_order_scheduled(self):
            self.gamestate.unschedule_order(self)

    def resume(self) -> None:
        if not self.gamestate.is_order_scheduled(self):
            self.gamestate.schedule_order_immediate(self)

    def register(self) -> None:
        self.gamestate.register_order(self)

    def unregister(self) -> None:
        self.gamestate.unregister_order(self)

    @abc.abstractmethod
    def act(self, dt:float) -> None:
        """ Performs one immediate tick's worth of action for this order """
        pass

class NullOrder(Order):
    @classmethod
    def create_null_order[T:"NullOrder"](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        return cls.create_order(*args, **kwargs)
    def act(self, dt:float) -> None:
        pass
