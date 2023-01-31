""" Interface event handling logic """

from typing import Mapping, Any

from stellarpunk import core, events, interface, narrative, util


class PlayerNotification(events.Action):
    def __init__(self, interface: interface.Interface) -> None:
        self.interface = interface

    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        if not all(
            k in action_args and isinstance(action_args[k], t) for k,t in [
                ("message", str),
            ]
        ):
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
        format_args = dict(event_args)
        format_args["_character"] = character
        message_str = action_args["message"].format(**format_args)
        self.interface.log_message(message_str)


class PlayerReceiveBroadcast(events.Action):
    def __init__(self, interface: interface.Interface) -> None:
        self.interface = interface

    def act(
        self,
        character: core.Character,
        event_type: int,
        event_context: Mapping[int, int],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        sender = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.MESSAGE_SENDER)]]
        assert isinstance(sender, core.Character)
        distance = util.distance(character.location.loc, sender.location.loc)
        self.interface.log_message(f'Bcast from {sender.address_str()} at {distance:.0f}m:\n{event_args["message"]}')


class PlayerReceiveMessage(events.Action):
    def __init__(self, interface: interface.Interface) -> None:
        self.interface = interface

    def act(
        self,
        character: core.Character,
        event_type: int,
        event_context: Mapping[int, int],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        sender = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.MESSAGE_SENDER)]]
        message = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.MESSAGE)]]
        assert isinstance(sender, core.Character)
        assert isinstance(message, core.Message)

        self.interface.log_message(f'Message from {sender.address_str()}:\n{message.subject}')


