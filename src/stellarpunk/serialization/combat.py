import io
import uuid
from typing import Any

from stellarpunk import core
from stellarpunk.core import combat
from stellarpunk.serialization import save_game, util as s_util

class TimedOrderTaskSaver(save_game.Saver[combat.TimedOrderTask]):
    def save(self, obj:combat.TimedOrderTask, f:io.IOBase) -> int:
        return s_util.uuid_to_f(obj.order.order_id, f)
    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> combat.TimedOrderTask:
        order_id = s_util.uuid_from_f(f)
        tot = combat.TimedOrderTask()
        load_context.register_post_load(tot, order_id)
        return tot
    def post_load(self, obj:combat.TimedOrderTask, load_context:save_game.LoadContext, context:Any) -> None:
        order_id:uuid.UUID = context
        order = load_context.gamestate.orders[order_id]
        assert(isinstance(order, core.Order))
        obj.order = order
        order.observe(obj)

#TODO: ThreatTracker
#TODO: PointDefenseEffect
#TODO: MissileOrder
#TODO: AttackOrder
#TODO: HuntOrder
#TODO: FleeOrder

