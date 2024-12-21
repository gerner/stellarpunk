import io
import abc
from typing import Any, Optional

from stellarpunk import core

from . import save_game, util as s_util, gamestate as s_gamestate

class OrderSaver[Order: core.Order](save_game.Saver[Order], abc.ABC):
    @abc.abstractmethod
    def _save_order(self, order:Order, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext) -> Order: ...

    def save(self, obj:Order, f:io.IOBase) -> int:
        return 0


    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Order:
        return self._load_order(f, load_context)
