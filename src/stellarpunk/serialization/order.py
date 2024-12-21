import io
import abc
import uuid
from typing import Any, Optional

from stellarpunk import core

from . import save_game, util as s_util, gamestate as s_gamestate

class OrderSaver[Order: core.Order](save_game.Saver[Order], abc.ABC):
    @abc.abstractmethod
    def _save_order(self, order:Order, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext) -> tuple[Order, Any]: ...
    @abc.abstractmethod
    def _post_load_order(self, order:Order, load_context:save_game.LoadContext, extra_context:Any) -> None: ...

    def save(self, order:Order, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.ship.entity_id, f)
        bytes_written += s_util.float_to_f(order.started_at, f)
        bytes_written += s_util.float_to_f(order.completed_at, f)
        bytes_written += s_util.float_to_f(order.init_eta, f)

        if order.parent_order:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(order.order_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        bytes_written += s_util.size_to_f(len(order.child_orders), f)
        for child in order.child_orders:
            bytes_written += s_util.uuid_to_f(child.order_id, f)

        #TODO: observers

        bytes_written += self._save_order(order, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Order:
        ship_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        completed_at = s_util.float_from_f(f)
        init_eta = s_util.float_from_f(f)
        has_parent = s_util.int_from_f(f, blen=1)
        parent_id:Optional[uuid.UUID] = None
        if has_parent:
            parent_id = s_util.uuid_from_f(f)
        count = s_util.size_from_f(f)
        child_ids:list[uuid.UUID] = []
        for i in range(count):
            child_id = s_util.uuid_from_f(f)
            child_ids.append(child_id)

        #TODO: observers

        order, extra_context = self._load_order(f, load_context)
        load_context.register_post_load(order, (ship_id, parent_id, child_ids, extra_context))
        return order

    def post_load(self, order:Order, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Optional[uuid.UUID], list[uuid.UUID], Any] = context
        ship_id, parent_id, child_ids, extra_context = context_data

        ship = load_context.gamestate.entities[ship_id]
        assert(isinstance(ship, core.Ship))
        order.initialize_order(ship)
        if parent_id:
            order.parent_order = load_context.gamestate.orders[parent_id]
        for child_id in child_ids:
            order.child_orders.append(load_context.gamestate.orders[child_id])

        self._post_load_order(order, load_context, extra_context)
