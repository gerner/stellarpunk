""" Core events and context for stellarpunk """

import enum
import numbers
from typing import Mapping, Any, Tuple

from . import core as ecore
from stellarpunk import narrative, core, dialog, util


POS_INF = (1<<64)-1


class Events(enum.IntEnum):
    START_GAME = enum.auto()
    NOTIFICATION = enum.auto()
    BROADCAST = enum.auto()
    MESSAGE = enum.auto()
    CONTACT = enum.auto()
    DOCKED = enum.auto()
    MINED = enum.auto()
    SOLD = enum.auto()
    BOUGHT = enum.auto()


class ContextKeys(enum.IntEnum):
    PLAYER = enum.auto()
    IS_PLAYER = enum.auto()
    SHIP = enum.auto()
    MESSAGE = enum.auto()
    MESSAGE_SENDER = enum.auto()
    MESSAGE_ID = enum.auto()
    CONTACTER = enum.auto()
    TARGET = enum.auto()
    RESOURCE = enum.auto()
    AMOUNT = enum.auto()
    AMOUNT_ON_HAND = enum.auto()
    TUTORIAL_GUY = enum.auto()
    TUTORIAL_TARGET_PLAYER = enum.auto()
    TUTORIAL_RESOURCE = enum.auto()
    TUTORIAL_AMOUNT_TO_MINE = enum.auto()
    TUTORIAL_AMOUNT_TO_TRADE = enum.auto()
    TUTORIAL_STARTED = enum.auto()
    TUTORIAL_SKIPPED = enum.auto()
    TUTORIAL_MINING_ARRIVE = enum.auto()
    TUTORIAL_MINED = enum.auto()
    TUTORIAL_DELIVERED = enum.auto()
    TUTORIAL_ASTEROID = enum.auto()

def send_message(gamestate:core.Gamestate, recipient:core.Character, message:core.Message) -> None:
        gamestate.trigger_event(
            [recipient],
            gamestate.event_manager.e(Events.MESSAGE),
            {
                gamestate.event_manager.ck(ContextKeys.MESSAGE_SENDER): util.uuid_to_u64(message.reply_to),
                gamestate.event_manager.ck(ContextKeys.MESSAGE_ID): message.message_id,
                gamestate.event_manager.ck(ContextKeys.MESSAGE): message.short_id_int(),
            },
        )


class IncAction(ecore.Action):
    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        if not self._required_keys([("flag", int)], action_args) or not self._optional_keys([("amount", int), ("amount_ref", int)], action_args):
            return False
        if ("amount" in action_args) == ("amount_ref" in action_args):
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
        current_value = character.context.get_flag(action_args["flag"])
        if "amount" in action_args:
            amount = action_args["amount"]
        else:
            amount = event_context.get(action_args["amount_ref"], 0)
        if current_value + amount < 0:
            new_value = 0
        elif current_value + amount > POS_INF:
            new_value = POS_INF
        else:
            new_value = current_value + amount
        character.context.set_flag(action_args["flag"], current_value+action_args["amount"])


class DecAction(ecore.Action):
    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        if not self._required_keys([("flag", int)], action_args) or not self._optional_keys([("amount", int), ("amount_ref", int)], action_args):
            return False
        if ("amount" in action_args) == ("amount_ref" in action_args):
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
        current_value = character.context.get_flag(action_args["flag"])
        if "amount" in action_args:
            amount = action_args["amount"]
        else:
            amount = event_context.get(action_args["amount_ref"], 0)
        if current_value - amount < 0:
            new_value = 0
        elif current_value - amount > POS_INF:
            new_value = POS_INF
        else:
            new_value = current_value - amount
        character.context.set_flag(action_args["flag"], new_value)


class BroadcastAction(ecore.Action):
    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        return self._required_keys([
                ("radius", numbers.Real),
                ("message_id", int),
                ("message", str),
            ],
            action_args,
        )

    def act(
        self,
        character: "core.Character",
        event_type: int,
        event_context: Mapping[int, int],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        assert character.location is not None
        assert character.location.sector is not None
        sector = character.location.sector
        loc = character.location.loc
        radius = action_args["radius"]
        message_id = action_args["message_id"]
        format_args = dict(event_args)
        format_args["_character"] = character
        message = action_args["message"].format(**format_args)

        loc = character.location.loc
        nearby_characters = list(
            x.captain for x in sector.spatial_point(loc, radius) if isinstance(x, core.CrewedSectorEntity) and x.captain is not None and x.captain != character
        )

        if len(nearby_characters) == 0:
            return

        self.gamestate.trigger_event(
            nearby_characters,
            self.gamestate.event_manager.e(Events.BROADCAST),
            {
                self.gamestate.event_manager.ck(ContextKeys.MESSAGE_SENDER): character.short_id_int(),
                self.gamestate.event_manager.ck(ContextKeys.MESSAGE_ID): message_id,
                self.gamestate.event_manager.ck(ContextKeys.SHIP): character.location.short_id_int(),
            },
            {"message": message},
        )


class MessageAction(ecore.Action):
    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        if not self._required_keys([
                ("message_id", int),
                ("subject", str),
                ("message", str),
                ("recipient", int),
            ],
            action_args,
        ):
            return False

        if not self._optional_keys([("sender", int)], action_args):
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

        if "sender" in action_args:
            sender_key = action_args["sender"]
            if sender_key in event_context:
                sender_id = event_context[action_args["sender"]]
            elif character.context.get_flag(sender_key) > 0:
                sender_id = character.context.get_flag(sender_key)
            else:
                raise ValueError(f'sender key {action_args["sender"]} not found in event or character context')
            sender = self.gamestate.entities_short[sender_id]
            assert isinstance(sender, core.Character)
        else:
            sender = character

        assert sender.location is not None
        assert sender.location.sector is not None
        sector = sender.location.sector
        message_id = action_args["message_id"]
        format_args = dict(event_args)
        format_args["_character"] = sender
        subject_str = action_args["subject"].format(**format_args)
        message_str = action_args["message"].format(**format_args)

        recipient_id = event_context[action_args["recipient"]] if action_args["recipient"] in event_context else sender.context.get_flag(action_args["recipient"])
        recipient = self.gamestate.entities_short[recipient_id]
        assert isinstance(recipient, core.Character)

        message = core.Message(message_id, subject_str, message_str, self.gamestate.timestamp, sender.entity_id, self.gamestate)

        send_message(self.gamestate, recipient, message)


def register_events(event_manager:ecore.EventManager) -> None:
    event_manager.register_events(Events)
    event_manager.register_context_keys(ContextKeys)
    event_manager.register_action(IncAction(), "inc")
    event_manager.register_action(DecAction(), "dec")
    event_manager.register_action(BroadcastAction(), "broadcast")
    event_manager.register_action(MessageAction(), "message")
