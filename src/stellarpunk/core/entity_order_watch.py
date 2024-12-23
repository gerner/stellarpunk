from typing import Optional

from stellarpunk.core.order import Order, OrderObserver
from stellarpunk.core.sector import Sector, SectorEntity, SectorEntityObserver

class EntityOrderWatch(OrderObserver, SectorEntityObserver):
    """ Watches a SectorEntity, cancels an order on destroy/migrate """
    def __init__(self, order:Order, target:SectorEntity) -> None:
        self.order:Optional[Order] = order
        self.order.observe(self)
        self.target:Optional[SectorEntity] = target
        self.target.observe(self)

    def order_complete(self, order:Order) -> None:
        if self.order is None:
            return
        assert order == self.order
        assert self.target
        self.target.unobserve(self)
        self.target = None
        self.order = None

    def order_cancel(self, order:Order) -> None:
        if self.order is None:
            return
        assert order == self.order
        assert self.target
        self.target.unobserve(self)
        self.target = None
        self.order = None

    def entity_destroyed(self, entity:SectorEntity) -> None:
        if self.target is None:
            return
        assert entity == self.target
        assert self.order
        self.order.cancel_order()
        self.target = None
        self.order = None

    def entity_migrated(self, entity:SectorEntity, from_sector:Sector, to_sector:Sector) -> None:
        if self.target is None:
            return
        assert entity == self.target
        assert self.order
        # TODO: do we always want to cancel on migrated? maybe optional?
        self.target.unobserve(self)
        self.order.cancel_order()
        self.target = None
        self.order = None
