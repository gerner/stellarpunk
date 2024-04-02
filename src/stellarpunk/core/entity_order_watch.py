from stellarpunk.core.order import Order, OrderObserver
from stellarpunk.core.sector import Sector
from stellarpunk.core.sector_entity import SectorEntity, SectorEntityObserver

class EntityOrderWatch(OrderObserver, SectorEntityObserver):
    """ Watches a SectorEntity, cancels an order on destroy/migrate """
    def __init__(self, order:Order, target:SectorEntity) -> None:
        self.order = order
        self.order.observe(self)
        self.target = target
        self.target.observe(self)

    def order_complete(self, order:Order) -> None:
        assert order == self.order
        self.target.unobserve(self)
        self.target = None
        self.order = None

    def order_cancel(self, order:Order) -> None:
        assert order == self.order
        self.target.unobserve(self)
        self.target = None
        self.order = None

    def entity_destroyed(self, entity:SectorEntity) -> None:
        assert entity == self.target
        self.order.cancel_order()
        self.target = None
        self.order = None

    def entity_migrated(self, entity:SectorEntity, from_sector:Sector, to_sector:Sector) -> None:
        assert entity == self.target
        # TODO: do we always want to cancel on migrated? maybe optional?
        self.target.unobserve(self)
        self.order.cancel_order()
        self.target = None
        self.order = None
