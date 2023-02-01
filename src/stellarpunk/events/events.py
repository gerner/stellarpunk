""" Core events and context for stellarpunk """

import enum
import numbers
from typing import TYPE_CHECKING, Mapping, Any, Tuple

from . import core as ecore
from stellarpunk import narrative, core, dialog


class Events(enum.IntEnum):
    START_GAME = enum.auto()
    NOTIFICATION = enum.auto()
    BROADCAST = enum.auto()
    MESSAGE = enum.auto()
    CONTACT = enum.auto()


class ContextKeys(enum.IntEnum):
    PLAYER = enum.auto()
    IS_PLAYER = enum.auto()
    SHIP = enum.auto()
    MESSAGE = enum.auto()
    MESSAGE_SENDER = enum.auto()
    MESSAGE_ID = enum.auto()
    CONTACTER = enum.auto()
    TUTORIAL_GUY = enum.auto()
    TUTORIAL = enum.auto()
    TUTORIAL_TARGET_PLAYER = enum.auto()
    TUTORIAL_STARTED = enum.auto()
    TUTORIAL_SKIPPED = enum.auto()


class BroadcastAction(ecore.Action):
    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        return all(
            k in action_args and isinstance(action_args[k], t) for k,t in [
                ("radius", numbers.Real),
                ("message_id", numbers.Real),
                ("message", str),
            ]
        )

    def act(
        self,
        character: "core.Character",
        event_type: int,
        event_context: Mapping[int, int],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        sector = character.location.sector
        assert sector is not None
        loc = character.location.loc
        radius = action_args["radius"]
        message_id = action_args["message_id"]
        format_args = dict(event_args)
        format_args["_character"] = character
        message = action_args["message"].format(**format_args)

        loc = character.location.loc
        nearby_characters = list(
            x.captain for x in sector.spatial_point(loc, radius) if x.captain is not None and x.captain != character
        )

        if len(nearby_characters) == 0:
            return

        self.gamestate.trigger_event(
            nearby_characters,
            ecore.e(Events.BROADCAST),
            {
                ecore.ck(ContextKeys.MESSAGE_SENDER): character.short_id_int(),
                ecore.ck(ContextKeys.MESSAGE_ID): message_id,
                ecore.ck(ContextKeys.SHIP): character.location.short_id_int(),
            },
            message=message,
        )


class MessageAction(ecore.Action):
    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        if not all(
            k in action_args and isinstance(action_args[k], t) for k,t in [
                ("message_id", numbers.Real),
                ("subject", str),
                ("message", str),
                ("recipient", str),
            ]
        ):
            return False

        if not action_args["recipient"].startswith("_ref:"):
            return False

        return True

    def act(
        self,
        character: "core.Character",
        event_type: int,
        event_context: Mapping[int, int],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:

        sector = character.location.sector
        assert sector is not None
        message_id = action_args["message_id"]
        format_args = dict(event_args)
        format_args["_character"] = character
        subject_str = action_args["subject"].format(**format_args)
        message_str = action_args["message"].format(**format_args)

        if "dialog" in action_args:
            reply_dialog = dialog.load_dialog(action_args["dialog"])
        else:
            reply_dialog = None

        recipient_id = event_context[action_args["recipient"]] if action_args["recipient"] in event_context else character.context.get_flag(action_args["recipient"])
        recipient = self.gamestate.entities_short[recipient_id]
        assert isinstance(recipient, core.Character)

        message = core.Message(message_id, subject_str, message_str, self.gamestate.timestamp, character, self.gamestate, reply_dialog=reply_dialog)

        self.gamestate.send_message(recipient, message)


def register_events() -> None:
    ecore.register_events(Events)
    ecore.register_context_keys(ContextKeys)
    ecore.register_action(BroadcastAction())
    ecore.register_action(MessageAction())

# TODO: should we not eagerly do this?
register_events()
