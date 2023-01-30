""" Core events and context for stellarpunk """

import enum
import numbers
from typing import TYPE_CHECKING, Mapping, Any, Tuple

from . import core as ecore
from stellarpunk import narrative, core


class Events(enum.IntEnum):
    START_GAME = enum.auto()
    DESTINATION_ARRIVE = enum.auto()
    BROADCAST = enum.auto()




class ContextKeys(enum.IntEnum):
    IS_PLAYER = enum.auto()
    SHIP = enum.auto()
    MESSAGE_SENDER = enum.auto()
    MESSAGE_ID = enum.auto()
    TUTORIAL_GUY = enum.auto()




class BroadcastEffect(core.Effect):
    def __init__(
        self,
        sender: core.Character,
        radius: float,
        message_id: int,
        message: str,
        expiration: float,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.sender = sender
        self.radius = radius
        self.message_id = message_id
        self.message = message
        self.expiration = expiration
        self.transmitted = False

    def _begin(self) -> None:
        self.gamestate.schedule_effect(self.expiration, self)

    def bbox(self) -> Tuple[float, float, float, float]:
        loc = self.sender.location.loc
        return (loc[0], loc[1], loc[0], loc[1])

    def is_complete(self) -> bool:
        return self.transmitted

    def act(self, dt: float) -> None:
        if self.sector != self.sender.location.sector:
            self.logger.warning(f'cancelling broadcast for character {self.sender.address_str()} who changed sector from {self.sector.short_id()}')
            self.cancel_effect()
        elif self.gamestate.timestamp >= self.expiration:
            loc = self.sender.location.loc
            nearby_characters = list(
                x.captain for x in self.sector.spatial_point(loc, self.radius) if x.captain is not None and x.captain != self.sender
            )

            self.gamestate.trigger_event(
                nearby_characters,
                ecore.e(Events.BROADCAST),
                narrative.context({
                    ecore.ck(ContextKeys.MESSAGE_SENDER): self.sender.short_id_int(),
                    ecore.ck(ContextKeys.MESSAGE_ID): self.message_id,
                    ecore.ck(ContextKeys.SHIP): self.sender.location.short_id_int(),
                }),
                self.sender,
                self.sender.location,
                message=self.message,
            )
            self.transmitted = True

            self.complete_effect()
        else:
            self.gamestate.schedule_effect(self.expiration, self)


class BroadcastAction(ecore.Action):
    def validate(self, action_args: Mapping[str, Any]) -> bool:
        return all(
            k in action_args and isinstance(action_args[k], t) for k,t in [
                ("radius", numbers.Real),
                ("message_id", numbers.Real),
                ("message", str),
                ("delay", numbers.Real),
            ]
        )

    def act(
        self,
        character: "core.Character",
        event_type: int,
        event_context: narrative.EventContext,
        entities: Mapping[int, "core.Entity"],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:

        sector = character.location.sector
        assert sector is not None
        loc = character.location.loc
        radius = action_args["radius"]
        message_id = action_args["message_id"]
        message = action_args["message"].format(**event_args)
        expiration = self.gamestate.timestamp + action_args["delay"]

        sector.add_effect(BroadcastEffect(
            character,
            radius,
            message_id,
            message,
            expiration,
            sector,
            self.gamestate,
        ))


def register_events() -> None:
    ecore.register_events(Events)
    ecore.register_context_keys(ContextKeys)
    ecore.register_action(BroadcastAction())

# TODO: should we not eagerly do this?
register_events()
