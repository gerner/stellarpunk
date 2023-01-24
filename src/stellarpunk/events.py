""" Manages Events """

import abc
import logging
import re
from typing import List, Any, Sequence, Dict, Tuple, Optional
from dataclasses import dataclass

from stellarpunk import core, util, dialog, predicates, config, task_schedule


class AbstractPlayerEventHandler:
    def handle_event(self, event: core.Event) -> None:
        pass


class AbstractEventManager:
    def tick(self) -> None:
        pass


class Action:
    def __init__(self, character: core.Character, event: core.Event, gamestate: core.Gamestate):
        self.character = character
        self.event = event
        self.gamestate = gamestate

    def act(self) -> None:
        pass

class BroadcastAction(Action):
    def __init__(
        self,
        sector: core.Sector,
        loc: Tuple[float, float],
        message_id: int,
        message: str,
        *args: Any,
        radius: float = 5e3,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sector = sector
        self.loc = loc
        self.radius = radius
        self.message_id = message_id
        self.message = message

    def act(self) -> None:
        nearby_characters = list(
            x.captain for x in self.sector.spatial_point(self.loc, self.radius) if x.captain is not None and x.captain != self.character
        )

        # TODO: this triggers many events, one for each character in range.
        # maybe we actually want to trigger a single event, but associate it
        # somehow with all of these potential characters.
        for receiver in nearby_characters:
            destination = self.event.get_entity(core.ContextKey.DESTINATION)
            self.gamestate.trigger_event(
                receiver,
                core.EventType.BROADCAST,
                {
                    core.ContextKey.MESSAGE_SENDER: self.character.short_id_int(),
                    core.ContextKey.MESSAGE_ID: self.message_id,
                    core.ContextKey.DESTINATION: self.event.context[core.ContextKey.DESTINATION],
                    core.ContextKey.SHIP: self.event.context[core.ContextKey.SHIP],
                },
                self.character,
                self.event.get_entity(core.ContextKey.DESTINATION),
                self.event.get_entity(core.ContextKey.SHIP),
                message=self.message,
            )


class EventManager(AbstractEventManager):
    def __init__(
        self,
        player_event_handler: Optional[AbstractPlayerEventHandler] = None
    ) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate:core.Gamestate = None # type: ignore[assignment]
        self.action_schedule:task_schedule.TaskSchedule[Action] = task_schedule.TaskSchedule()
        self.player_event_handler = player_event_handler or AbstractPlayerEventHandler()

    def initialize(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate

    def tick(self) -> None:
        # check for relevant events and process them
        while len(self.gamestate.events) > 0:
            event = self.gamestate.events.popleft()

            self.logger.debug(f'event {str(core.EventType(event.event_type))} received by {event.character.short_id()}')

            if event.character.to_context().get(core.ContextKey.IS_PLAYER, 0) == 1:
                # player events get handled separately
                self.player_event_handler.handle_event(event)
                continue

            # HACK: just to get event model POC for comms chatter
            if event.event_type == core.EventType.APPROACH_DESTINATION:
                destination = event.get_entity(core.ContextKey.DESTINATION)
                if not isinstance(destination, core.Station):
                    continue
                assert destination.sector is not None
                ship = event.get_entity(core.ContextKey.SHIP)
                assert isinstance(ship, core.Ship)
                assert ship.sector is not None

                self.action_schedule.push_task(
                    self.gamestate.timestamp+0.5,
                    BroadcastAction(
                        ship.sector, (ship.loc[0], ship.loc[1]),
                        0,
                        f'Anyone at {destination.short_id()}, this is {event.character.name} of {ship.short_id()}. I\'m headed in, who want to grab a drink?',
                        event.character, event, self.gamestate
                    )
                )
            elif event.event_type == core.EventType.BROADCAST:
                if event.context[core.ContextKey.MESSAGE_ID] == 0:
                    destination = event.get_entity(core.ContextKey.DESTINATION)
                    assert isinstance(destination, core.Station)
                    assert destination.sector is not None
                    ship = event.get_entity(core.ContextKey.SHIP)
                    assert isinstance(ship, core.Ship)

                    self.action_schedule.push_task(
                        self.gamestate.timestamp+0.5,
                        BroadcastAction(
                            destination.sector, (destination.loc[0], destination.loc[1]),
                            1,
                            f'{ship.short_id()} this is {event.character.name} of {event.character.location.short_id()} if you\'re buying, I\'m in.',
                            event.character, event, self.gamestate
                        )
                    )

        for action in self.action_schedule.pop_current_tasks(self.gamestate.timestamp):
            action.act()


class DialogManager:
    def __init__(self, dialog:dialog.DialogGraph, gamestate:core.Gamestate, player:core.Player) -> None:
        self.dialog = dialog
        self.current_id = dialog.root_id
        self.gamestate = gamestate
        self.player = player

    @property
    def node(self) -> dialog.DialogNode:
        return self.dialog.nodes[self.current_id]

    @property
    def choices(self) -> Sequence[dialog.DialogChoice]:
        return self.dialog.nodes[self.current_id].choices

    def do_node(self) -> None:
        #for node_event_id in self.dialog.nodes[self.current_id].event_id:
        #    self.player.set_flag(node_event_id, self.gamestate.timestamp)
        pass

    def choose(self, choice:dialog.DialogChoice) -> None:
        if choice not in self.choices:
            raise ValueError(f'given choice is not current node {self.current_id} choices')

        #for event_id in choice.event_id:
        #    self.player.set_flag(event_id, self.gamestate.timestamp)
        self.current_id = choice.node_id
