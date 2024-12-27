import io
import abc
import uuid
import collections
from typing import Any, Optional

from stellarpunk import core, util

from . import save_game, util as s_util, gamestate as s_gamestate

class OrderSaver[Order: core.Order](save_game.Saver[Order], abc.ABC):
    @abc.abstractmethod
    def _save_order(self, order:Order, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[Order, Any]: ...
    def _post_load_order(self, order:Order, load_context:save_game.LoadContext, extra_context:Any) -> None:
        pass

    def save(self, order:Order, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(order.order_id, f)
        bytes_written += s_util.uuid_to_f(order.ship.entity_id, f)
        bytes_written += s_util.float_to_f(order.started_at, f)
        bytes_written += s_util.float_to_f(order.completed_at, f)
        bytes_written += s_util.float_to_f(order.init_eta, f)

        if order.parent_order:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(order.order_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        bytes_written += s_util.debug_string_w("child orders", f)
        bytes_written += s_util.size_to_f(len(order.child_orders), f)
        for child in order.child_orders:
            bytes_written += s_util.uuid_to_f(child.order_id, f)

        if self.save_game.debug:
            bytes_written += s_util.debug_string_w("observers", f)
            bytes_written += s_util.str_uuids_to_f(list((util.fullname(x), x.observer_id) for x in order.observers), f)

        bytes_written += s_util.debug_string_w("type specific", f)
        bytes_written += self._save_order(order, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Order:
        s_util.debug_string_r("basic fields", f)
        order_id = s_util.uuid_from_f(f)
        ship_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        completed_at = s_util.float_from_f(f)
        init_eta = s_util.float_from_f(f)
        has_parent = s_util.int_from_f(f, blen=1)
        parent_id:Optional[uuid.UUID] = None
        if has_parent:
            parent_id = s_util.uuid_from_f(f)

        s_util.debug_string_r("child orders", f)
        count = s_util.size_from_f(f)
        child_ids:list[uuid.UUID] = []
        for i in range(count):
            child_id = s_util.uuid_from_f(f)
            child_ids.append(child_id)

        observer_ids:list[tuple[str, uuid.UUID]] = []
        if load_context.debug:
            s_util.debug_string_r("observers", f)
            observer_ids = s_util.str_uuids_from_f(f)

        s_util.debug_string_r("type specific", f)
        order, extra_context = self._load_order(f, load_context, order_id)
        load_context.gamestate.register_order(order)
        load_context.register_post_load(order, (ship_id, parent_id, child_ids, extra_context))

        if load_context.debug:
            load_context.register_sanity_check(order, observer_ids)

        return order

    def post_load(self, order:Order, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Optional[uuid.UUID], list[uuid.UUID], Any] = context
        ship_id, parent_id, child_ids, extra_context = context_data

        ship = load_context.gamestate.entities[ship_id]
        assert(isinstance(ship, core.Ship))
        order.initialize_order(ship)
        if parent_id:
            parent_order = load_context.gamestate.orders[parent_id]
            assert(isinstance(parent_order, core.Order))
            order.parent_order = parent_order
        for child_id in child_ids:
            child_order = load_context.gamestate.orders[child_id]
            assert(isinstance(child_order, core.Order))
            order.child_orders.append(child_order)

        self._post_load_order(order, load_context, extra_context)

    def sanity_check(self, order:Order, load_context:save_game.LoadContext, context:Any) -> None:
        observer_ids:list[tuple[str, uuid.UUID]] = context

        # make sure all the observers we had when saving are back
        saved_observer_counts:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for observer_id in observer_ids:
            saved_observer_counts[observer_id] += 1
        loaded_observer_counts:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for observer in order.observers:
            loaded_observer_counts[(util.fullname(observer), observer.observer_id)] += 1
        saved_observer_counts.subtract(loaded_observer_counts)
        non_zero_observers = {observer_id: count for observer_id, count in saved_observer_counts.items() if count != 0}
        assert(non_zero_observers == {})

class NullOrderSaver(OrderSaver[core.NullOrder]):
    def _save_order(self, order:core.NullOrder, f:io.IOBase) -> int:
        return 0

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[core.NullOrder, Any]:
        return (core.NullOrder(load_context.gamestate, _check_flag=True, order_id=order_id), None)

