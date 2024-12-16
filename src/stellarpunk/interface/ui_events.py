""" Interface event handling logic """

import logging
from typing import Mapping, MutableMapping, Any

from stellarpunk import core, events, interface, narrative, util, dialog
from stellarpunk.interface import comms

logger = logging.getLogger(__name__)

class DialogAction(events.Action):
    def __init__(self, interface: interface.Interface, event_manager: events.EventManager) -> None:
        self.interface = interface
        self.event_manager = event_manager

    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        #if not all(
        #    k in action_args and isinstance(action_args[k], t) for k,t in [
        #        ("dialog_id", str),
        #    ]
        #):
        #    return False
        return True

    def act(
        self,
        character: "core.Character",
        event_type: int,
        event_context: Mapping[int, int],
        event_args: MutableMapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        contacter = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.CONTACTER)]]
        assert isinstance(contacter, core.Character)
        #TODO: handle other characters contacting the player and not just
        # player initiating the contact
        assert contacter == self.interface.player.character

        if "dialog_id" not in action_args:
            logger.info(f'{core.Gamestate.gamestate.timestamp} no dialog: {character} {action_args}')
            no_response_view = comms.NoResponseView(character, self.gamestate, self.interface)
            self.interface.open_view(no_response_view, deactivate_views=True)
        else:
            dialog_id = action_args["dialog_id"]
            logger.info(f'{core.Gamestate.gamestate.timestamp} dialog: {character} {action_args}')
            comms_view = comms.CommsView(
                events.DialogManager(dialog.load_dialog(dialog_id), self.gamestate, self.event_manager, contacter, character),
                character,
                self.gamestate,
                self.interface,
            )
            self.interface.open_view(comms_view, deactivate_views=True)
            event_args["dialog"] = True


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
        event_args: MutableMapping[str, Any],
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
        event_args: MutableMapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        sender = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.MESSAGE_SENDER)]]
        assert isinstance(sender, core.Character)
        assert sender.location
        assert character.location
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
        event_args: MutableMapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        sender = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.MESSAGE_SENDER)]]
        message = self.gamestate.entities_short[event_context[events.ck(events.ContextKeys.MESSAGE)]]
        assert isinstance(sender, core.Character)
        assert isinstance(message, core.Message)

        self.interface.player.messages[message.entity_id] = message
        self.interface.log_message(f'Message {message.short_id()} from {sender.address_str()}:\n{message.subject}')


