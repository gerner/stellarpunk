""" Comms view between a character and the player """

import abc
from typing import Any, Optional, Sequence, Collection, Deque, Callable
import curses
import textwrap
import collections
import enum
import time

import numpy as np
import dtmf # type: ignore

from stellarpunk import core, interface, config, dialog, events
from stellarpunk.interface import ui_util

class AnimationSequence:
    def __init__(self, callback:Optional[Callable[[], None]]=None):
        self.callback = callback
        self.start_time = -1.0

    def _initialize(self, now:float) -> None:
        pass

    @abc.abstractmethod
    def _animate(self, now:float) -> bool: ...

    def _finish(self) -> None:
        pass

    def _flush(self) -> None:
        pass

    def animate(self, now:float) -> bool:
        if self.start_time < 0:
            self.start_time = now
            self._initialize(now)

        if self._animate(now):
            self._finish()
            if self.callback is not None:
                self.callback()
            return True
        return False

    def flush(self) -> None:
        self._flush()
        self._finish()
        if self.callback is not None:
            self.callback()

class DialAnimation(AnimationSequence):
    def __init__(self, number_str:str, mixer:interface.AbstractMixer, canvas:interface.Canvas, *args:Any, end_str:str="\n\n", **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.mixer = mixer
        self.canvas = canvas
        self.number_str = number_str
        self.duration = 0.0
        self.end_str = end_str
        self.channel = -2

    def _initialize(self, now:float) -> None:
        self.channel = self.mixer.play_sample(
            np.hstack((
                ui_util.dtmf_sample(dtmf.model.String([dtmf.model.Tone("dial"), dtmf.model.Pause()]), self.mixer.sample_rate, mark_duration=0.6, space_duration=0.1, pause_duration=0.05),
                ui_util.dtmf_sample(self.number_str, self.mixer.sample_rate),
            )),
        )
        self.duration = 0.75 + 0.06*len(self.number_str)
        self.canvas.window.addstr(f'Dialing {self.number_str}...')
        self.canvas.noutrefresh(0, 0)

    def _animate(self, now:float) -> bool:
        return now > self.start_time + self.duration

    def _finish(self) -> None:
        if self.end_str:
            self.canvas.window.addstr(self.end_str)

    def _flush(self) -> None:
        self.mixer.halt_channel(self.channel)

class RingingAnimation(AnimationSequence):
    def __init__(self, mixer:interface.AbstractMixer, canvas:interface.Canvas, *args:Any, end_str:str="\n\n", **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.mixer = mixer
        self.canvas = canvas
        self.end_str = end_str
        self.channel = -2
        self.last_dot = 0.0
        self.dot_interval = 1.5

    def _initialize(self, now:float) -> None:
        self.channel = self.mixer.play_sample(
            ui_util.dtmf_sample(dtmf.model.String([dtmf.model.Tone("ringing"), dtmf.model.Tone("ringing")]), self.mixer.sample_rate, mark_duration=0.75, space_duration=0.75, pause_duration=0.1),
            loops=-1
        )

    def _animate(self, now:float) -> bool:
        if now-self.last_dot > self.dot_interval:
            self.canvas.window.addstr(f'.')
            self.last_dot = now
            self.canvas.noutrefresh(0, 0)
        return False

    def _finish(self) -> None:
        if self.end_str:
            self.canvas.window.addstr(self.end_str)
            self.canvas.noutrefresh(0, 0)

    def _flush(self) -> None:
        self.mixer.halt_channel(self.channel)

class DialogKeyFrame(AnimationSequence):
    def __init__(self, canvas:interface.Canvas, l_padding:str, text:str, chars_per_sec:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.l_padding = l_padding
        self.text = text
        self.canvas:interface.Canvas = canvas
        self.chars_per_sec:float = chars_per_sec
        self.position:int = 0

    def flush(self) -> None:
        if self.start_time < 0:
            self.canvas.window.addstr(self.l_padding)
        self.canvas.window.addstr(self.text[self.position:])
        self.canvas.noutrefresh(0, 0)

    def _initialize(self, now:float) -> None:
        if self.chars_per_sec > 0:
            self.canvas.window.addstr(self.l_padding)

    def _animate(self, now:float) -> bool:
        # we know what we're animating (dialog or response)
        # we know how far we are through animating it
        # we know when it started and when now currently is
        # we just need to "catch up" with now
        if self.chars_per_sec < 0:
            self.canvas.window.addstr(self.l_padding+self.text)
            self.canvas.noutrefresh(0, 0)
            return True

        secs = now - self.start_time
        end_position = int(secs * self.chars_per_sec)
        if end_position - self.position > 0:
            self.canvas.window.addstr(self.text[self.position:end_position])
            self.canvas.noutrefresh(0, 0)
            self.position = end_position

        return self.position >= len(self.text)

class DialogPause(AnimationSequence):
    def __init__(self, pause_length:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.pause_length = pause_length

    def _animate(self, now:float) -> bool:
        return now-self.start_time >= self.pause_length

class NoResponseView(interface.GameView):
    def __init__(self, character:core.Character, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.character = character
        self.dialog_pad:interface.Canvas = None # type: ignore[assignment]
        self.animation_queue:Deque[AnimationSequence] = collections.deque()

        self.channel:Optional[int] = None

        self.padding = 5

    def _flush_animation_queue(self) -> None:
        while len(self.animation_queue) > 0:
            self.animation_queue.popleft().flush()

    def initialize(self) -> None:
        # we want this to look like CommsView, so we force pause here
        self.gamestate.force_pause(self)
        self.interface.reinitialize_screen(name="Comms")

        dph = self.interface.viewscreen.height-self.padding*2
        dpw = self.interface.viewscreen.width-self.padding*2
        self.dialog_pad = interface.Canvas(
            curses.newpad(dph, dpw),
            dph,
            dpw,
            self.interface.viewscreen.y+self.padding,
            self.interface.viewscreen.x+self.padding,
            self.interface.aspect_ratio,
        )
        self.dialog_pad.window.scrollok(True)

        number_str = "".join(list(f'{oct(x)[2:]:0>4}' for x in self.character.entity_id.bytes[0:4]))
        self.animation_queue.append(DialAnimation(number_str, self.interface.mixer, self.dialog_pad, end_str=""))
        self.animation_queue.append(RingingAnimation(self.interface.mixer, self.dialog_pad, end_str="\nNo Answer.\n\n"))

    def termiante(self) -> None:
        self.gamestate.force_unpause(self)

    def key_list(self) -> Collection[interface.KeyBinding]:
        if len(self.animation_queue) > 0:
            return [self.bind_key(ord("\r"), self._flush_animation_queue)]
        else:
            return [self.bind_key(ord("\r"), lambda: self.interface.close_view(self))]
    def update_display(self) -> None:
        # handle "animation"
        while len(self.animation_queue) > 0 and self.animation_queue[0].animate(time.time()):
            self.animation_queue.popleft()

class CommsView(interface.GameView):
    def __init__(self, dialog_manager:events.DialogManager, speaker:core.Character, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # width we use to draw a portrait (and other info) for the speaker
        self.info_width = config.Settings.interface.CommsView.info_width
        self.padding = 5

        # width we use for the actual dialog (excluding the gutter space for
        # speaker info)
        self.dialog_width = 0

        self.response_indent = self.info_width+16
        self.response_width = 64

        # where the dialog plays out
        self.dialog_pad:interface.Canvas = None # type: ignore[assignment]

        #TODO: do we care about multiple speakers?
        self.speaker:core.Character = speaker
        self.last_character:Optional[core.Character] = None

        # manager to keep track and drive biz logic for the current dialog
        self.dialog_manager = dialog_manager

        # characters per second
        self.animation_speed:float = config.Settings.interface.CommsView.animation_speed
        self.pause_time:float = config.Settings.interface.CommsView.pause_time
        self.animation_queue:Deque[AnimationSequence] = collections.deque()

    def _addstr(self, l_padding:str, text:str, chars_per_sec:Optional[float]=None) -> None:
        if chars_per_sec is None:
            chars_per_sec = self.animation_speed
        self.animation_queue.append(DialogKeyFrame(
            self.dialog_pad, l_padding, text, chars_per_sec,
        ))

    def _flush_animation_queue(self) -> None:
        while len(self.animation_queue) > 0:
            self.animation_queue.popleft().flush()

    def initialize(self) -> None:
        # we don't want the dialog to have race conditions with rest of game
        # so we force pause while this view is up
        self.gamestate.force_pause(self)
        self.interface.reinitialize_screen(name="Comms")

        dph = self.interface.viewscreen.height-self.padding*2
        dpw = self.interface.viewscreen.width-self.padding*2
        self.dialog_pad = interface.Canvas(
            curses.newpad(dph, dpw),
            dph,
            dpw,
            self.interface.viewscreen.y+self.padding,
            self.interface.viewscreen.x+self.padding,
            self.interface.aspect_ratio,
        )
        self.dialog_pad.window.scrollok(True)

        self.dialog_width = dpw - self.info_width - self.padding

        self.response_width = 64
        self.response_indent = dpw - self.response_width

        self.interface.log_message(f'connection established with {self.speaker.short_id()}')

        number_str = "".join(list(f'{oct(x)[2:]:0>4}' for x in self.speaker.entity_id.bytes[0:4]))
        self.animation_queue.append(DialAnimation(number_str, self.interface.mixer, self.dialog_pad,
            lambda: self.handle_dialog_node(self.dialog_manager.node)
        ))

    def terminate(self) -> None:
        self.interface.log_message("connection closed.")
        self.gamestate.force_unpause(self)

    def handle_dialog_node(self, node:dialog.DialogNode) -> None:
        self.dialog_manager.do_node()
        self.add_message(self.speaker, node.text)
        self.add_responses(node)

    def handle_dialog_response(self, choice:dialog.DialogChoice) -> None:
        self.dialog_manager.choose(choice)
        self.add_player_message(choice.text)
        self.handle_dialog_node(self.dialog_manager.node)

    def add_responses(self, node:dialog.DialogNode) -> None:
        responses = node.choices
        left_padding = " " * self.response_indent

        if len(responses) == 0:
            assert node.terminal
            self._addstr(left_padding, 'press <ENTER> to close\n\n', -1)
        elif len(responses) == 1 and responses[0].text == "":
            self._addstr(left_padding, 'press <ENTER> to continue\n\n', -1)
        else:
            # indented to the right
            response_lines = [
                textwrap.wrap(
                    f' {i}: {response.text}',
                    width=self.response_width,
                    subsequent_indent=" "*4,
                ) for i, response in enumerate(responses, start=1)
            ]

            self._addstr(left_padding, f'Respond:\n', -1)
            for response in response_lines:
                for line in response:
                    self._addstr(left_padding, f'{line}\n', -1)
            self._addstr("", "\n", -1)

        self.dialog_pad.noutrefresh(0, 0)

    def add_message(self, character:core.Character, text:str) -> None:
        # message is left justified and indented to the right of the character
        # portrait

        assert character.location

        dialog_lines = textwrap.wrap(text, width=self.dialog_width)

        if character != self.last_character:
            portrait_lines = list(f'{x:^{self.info_width}}' for x in character.portrait.text)
            portrait_lines.extend([
                "",
                f'{character.name:^{self.info_width}}',
                f'{character.location.address_str():^{self.info_width}}',
            ])
            self.last_character = character
        else:
            portrait_lines = []

        # pad dialog or portrait with empty strings so they both have the same
        # number of lines so we can spit them out together
        if len(dialog_lines) < len(portrait_lines):
            dialog_lines.extend([""]*(len(portrait_lines)-len(dialog_lines)))
        elif len(portrait_lines) < len(dialog_lines):
            portrait_lines.extend([" "*self.info_width]*(len(dialog_lines)-len(portrait_lines)))

        portrait_padding_l = " " * ((self.info_width-self.info_width)//2+1)
        portrait_padding_r = " " * (self.info_width-self.info_width-len(portrait_padding_l)+1)

        assert len(dialog_lines) == len(portrait_lines)

        self.animation_queue.append(DialogPause(0.4))
        for (p,d) in zip(portrait_lines, dialog_lines):
            self._addstr(portrait_padding_l+p+portrait_padding_r, d)
            self._addstr("", "\n", -1)

        self._addstr("", "\n", -1)
        self.dialog_pad.noutrefresh(0, 0)

    def add_player_message(self, text:str) -> None:
        dialog_lines = textwrap.wrap(text, width=self.dialog_width)

        for d in dialog_lines:
            l_padding = " "*(self.dialog_width+self.info_width+self.padding-len(d))
            self._addstr(l_padding, f'{d}\n')

        self._addstr("", "\n", -1)
        self.dialog_pad.noutrefresh(0, 0)

    def choose_dialog_option(self, i:int) -> None:
        choice = self.dialog_manager.choices[i]
        self.logger.debug(f'pressed {i} corresponding to "{choice.text}" -> {choice.node_id}')
        self.handle_dialog_response(choice)

    def bind_dialog_option_key(self, key:int, i:int) -> interface.KeyBinding:
        return self.bind_key(key, lambda: self.choose_dialog_option(i))

    def key_list(self) -> Collection[interface.KeyBinding]:
        if len(self.animation_queue) > 0:
            return [self.bind_key(ord("\r"), self._flush_animation_queue)]
        else:
            keys = []
            if self.dialog_manager.node.terminal:
                keys.append(self.bind_key(ord("\r"), lambda: self.interface.close_view(self)))
            elif len(self.dialog_manager.choices) == 1 and self.dialog_manager.choices[0].text == "":
                keys.append(self.bind_dialog_option_key(ord("\r"), 0))
            else:
                keys.extend([
                        self.bind_dialog_option_key(ord(str(i+1)), i)
                        for i in range(len(self.dialog_manager.choices))
                ])
            return keys

    def update_display(self) -> None:

        # handle "animation" of the dialog
        while len(self.animation_queue) > 0 and self.animation_queue[0].animate(time.time()):
            self.animation_queue.popleft()

