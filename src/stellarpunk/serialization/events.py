import io
import uuid
from typing import Any, Union

from stellarpunk import events, core, util, narrative
from stellarpunk.serialization import save_game, util as s_util

from . import save_game

class EventStateSaver(save_game.Saver[events.EventState]):
    def _load_sanity_check(self, f:io.IOBase) -> None:
        res = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.int_from_f)
        rcs = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.int_from_f)
        actions = s_util.fancy_dict_from_f(f, s_util.int_from_f, s_util.from_len_pre_f)

        event_types = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.int_from_f)
        context_keys = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.int_from_f)
        action_ids = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.int_from_f)

        # check these are subsets of the registration info in event_manager
        # it's ok for new stuff to be added, but these spaces must all be
        # consistent
        assert(res.items() <= dict((util.fullname(k),v) for k,v in self.save_game.event_manager.RegisteredEventSpaces.items()).items())
        assert(rcs.items() <= dict((util.fullname(k),v) for k,v in self.save_game.event_manager.RegisteredContextSpaces.items()).items())
        assert(actions.items() <= dict((k,util.fullname(v)) for k,v in self.save_game.event_manager.actions.items()).items())

        assert(event_types.items() <= self.save_game.event_manager.event_types.items())
        assert(context_keys.items() <= self.save_game.event_manager.context_keys.items())
        assert(action_ids.items() <= self.save_game.event_manager.action_ids.items())

    def _save_event(self, event:narrative.Event, f:io.IOBase) -> int:
        def uint64_to_f(x:int, f:io.IOBase) -> int:
            return s_util.int_to_f(x, f, blen=8)

        bytes_written = 0
        bytes_written += s_util.int_to_f(event.event_type, f)
        bytes_written += s_util.fancy_dict_to_f(dict(event.event_context), f, uint64_to_f, uint64_to_f)
        # no need to serialize entity context here, each entity serializes
        # its own context
        bytes_written += s_util.fancy_dict_to_f(event.args, f, s_util.to_len_pre_f, s_util.primitive_to_f)
        return bytes_written

    def _load_event(self, f:io.IOBase) -> tuple[int, dict[int, int], dict[str, Union[int,float,str,bool]]]:
        def uint64_from_f(f:io.IOBase) -> int:
            return s_util.int_from_f(f, blen=8)

        # event
        event_type = s_util.int_from_f(f)
        event_context = s_util.fancy_dict_from_f(f, uint64_from_f, uint64_from_f)
        # no need to serialize entity context here, each entity
        # deserializes its own context
        event_args = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.primitive_from_f)
        return (event_type, event_context, event_args)

    def _save_action(self, action:narrative.Action, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.int_to_f(action.action_id, f)
        bytes_written += s_util.uuid_to_f(action.character_candidate.data.entity_id, f)
        bytes_written += s_util.fancy_dict_to_f(action.args, f, s_util.to_len_pre_f, s_util.primitive_to_f)
        return bytes_written

    def _load_action(self, f:io.IOBase) -> tuple[int, uuid.UUID, dict[str, Union[int,float,str,bool]]]:
            action_id = s_util.int_from_f(f)
            candidate_id = s_util.uuid_from_f(f)
            action_args = s_util.fancy_dict_from_f(f, s_util.from_len_pre_f, s_util.primitive_from_f)

            return (action_id, candidate_id, action_args)


    def save(self, event_state:events.EventState, f:io.IOBase) -> int:
        bytes_written = 0

        # debug info on event, context keys, actions for error checking
        bytes_written += s_util.debug_string_w("event space", f)
        bytes_written += s_util.fancy_dict_to_f(dict((util.fullname(k),v) for k,v in self.save_game.event_manager.RegisteredEventSpaces.items()), f, s_util.to_len_pre_f, s_util.int_to_f)
        bytes_written += s_util.fancy_dict_to_f(dict((util.fullname(k),v) for k,v in self.save_game.event_manager.RegisteredContextSpaces.items()), f, s_util.to_len_pre_f, s_util.int_to_f)
        bytes_written += s_util.fancy_dict_to_f(dict((k,util.fullname(v)) for k,v in self.save_game.event_manager.actions.items()), f, s_util.int_to_f, s_util.to_len_pre_f)

        bytes_written += s_util.fancy_dict_to_f(self.save_game.event_manager.event_types, f, s_util.to_len_pre_f, s_util.int_to_f)
        bytes_written += s_util.fancy_dict_to_f(self.save_game.event_manager.context_keys, f, s_util.to_len_pre_f, s_util.int_to_f)
        bytes_written += s_util.fancy_dict_to_f(self.save_game.event_manager.action_ids, f, s_util.to_len_pre_f, s_util.int_to_f)

        # event queue
        bytes_written += s_util.debug_string_w("event queue", f)
        bytes_written += s_util.size_to_f(len(event_state.event_queue), f)
        for event, candidates in event_state.event_queue:
            bytes_written += self._save_event(event, f)
            bytes_written += s_util.uuids_to_f(list(x.data.entity_id for x in candidates), f)

        # action schedule
        bytes_written += s_util.debug_string_w("action schedule", f)
        bytes_written += s_util.size_to_f(event_state.action_schedule.size(), f)
        for timestamp, (event, action) in event_state.action_schedule:
            bytes_written += s_util.float_to_f(timestamp, f)
            #TODO: we might be creating a lot of copies of this event, but the
            # underlying event_state only had one copy of the event for all
            # actions associated with it. should we create just one and then
            # have a reference to that one copy?
            bytes_written += self._save_event(event, f)
            bytes_written += self._save_action(action, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> events.EventState:
        # debug info for event, context keys, actions, for error checking
        s_util.debug_string_r("event space", f)
        self._load_sanity_check(f)

        # event queue (partial, we'll fully materialize in post_load)
        s_util.debug_string_r("event queue", f)
        loaded_events:list[tuple[tuple[int, dict[int, int], dict[str, Union[int,float,str,bool]]], list[uuid.UUID]]] = []
        count = s_util.size_from_f(f)
        for i in range(count):
            loaded_event = self._load_event(f)
            candidates:list[uuid.UUID] = s_util.uuids_from_f(f)

            loaded_events.append((loaded_event, candidates))

        # action schedule (partial, we'll fully materialize in post_load)
        s_util.debug_string_r("action schedule", f)
        loaded_actions:list[tuple[float, tuple[int, dict[int, int], dict[str, Union[int,float,str,bool]]], tuple[int, uuid.UUID, dict[str, Union[int,float,str,bool]]]]] = []
        count = s_util.size_from_f(f)
        for i in range(count):
            timestamp = s_util.float_from_f(f)
            loaded_event = self._load_event(f)
            loaded_action = self._load_action(f)
            loaded_actions.append((timestamp, loaded_event, loaded_action))

        event_state = events.EventState()
        load_context.register_post_load(event_state, (loaded_events, loaded_actions))

        return event_state

    def post_load(self, event_state:events.EventState, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[
            list[tuple[tuple[int, dict[int, int], dict[str, Union[int,float,str,bool]]], list[uuid.UUID]]],
            list[tuple[float, tuple[int, dict[int, int], dict[str, Union[int,float,str,bool]]], tuple[int, uuid.UUID, dict[str, Union[int,float,str,bool]]]]]
        ] = context
        events, actions = context_data

        for (event_type, event_context, event_args), candidates in events:
            characters:list[core.Character] = []
            for entity_id in candidates:
                entity = load_context.gamestate.entities[entity_id]
                assert(isinstance(entity, core.Character))
                characters.append(entity)
            event_state.event_queue.append((
                narrative.Event(
                    event_type,
                    event_context,
                    load_context.gamestate.entity_context_store,
                    event_args,
                ),
                [narrative.CharacterCandidate(c.context, c) for c in characters]
            ))

        for timestamp, (event_type, event_context, event_args), (action_id, candidate_id, action_args) in actions:
            event = narrative.Event(
                event_type,
                event_context,
                load_context.gamestate.entity_context_store,
                event_args,
            )
            c = load_context.gamestate.entities[candidate_id]
            assert(isinstance(c, core.Character))
            action = narrative.Action(
                action_id,
                narrative.CharacterCandidate(c.context, c),
                action_args
            )

            event_state.action_schedule.push_task(
                timestamp,
                (event, action)
            )


