""" Comms view between a character and the player """

from typing import Any, Optional, Sequence, Collection, Deque
import curses
import textwrap
import collections

from stellarpunk import core, interface, config, dialog, events
from stellarpunk.interface import ui_utils

class AnimationSequence:
    def animate(self, now:float) -> bool:
        pass

    def flush(self) -> None:
        pass

class DialogKeyFrame(AnimationSequence):
    def __init__(self, canvas:interface.Canvas, l_padding:str, text:str, chars_per_sec:float) -> None:
        self.start_time:float = -1
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

    def animate(self, now:float) -> bool:
        # we know what we're animating (dialog or response)
        # we know how far we are through animating it
        # we know when it started and when now currently is
        # we just need to "catch up" with now
        if self.chars_per_sec < 0:
            self.canvas.window.addstr(self.l_padding+self.text)
            self.canvas.noutrefresh(0, 0)
            return True

        if self.start_time < 0:
            self.start_time = now
            self.canvas.window.addstr(self.l_padding)

        secs = now - self.start_time
        end_position = int(secs * self.chars_per_sec)
        if end_position - self.position > 0:
            self.canvas.window.addstr(self.text[self.position:end_position])
            self.canvas.noutrefresh(0, 0)
            self.position = end_position

        return self.position >= len(self.text)

class DialogPause(AnimationSequence):
    def __init__(self, pause_length:float) -> None:
        self.end_time:float = -1
        self.pause_length = pause_length

    def animate(self, now:float) -> bool:
        if self.end_time < 0:
            self.end_time = now + self.pause_length

        return now >= self.end_time

class CommsView(interface.View):
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
        self.interface.reinitialize_screen(name="Comms")

        dph = self.interface.viewscreen_height-self.padding*2
        dpw = self.interface.viewscreen_width-self.padding*2
        self.dialog_pad = interface.Canvas(
            curses.newpad(dph, dpw),
            dph,
            dpw,
            self.interface.viewscreen_y+self.padding,
            self.interface.viewscreen_x+self.padding,
        )
        self.dialog_pad.window.scrollok(True)

        self.dialog_width = dpw - self.info_width - self.padding

        self.response_width = 64
        self.response_indent = dpw - self.response_width

        self.handle_dialog_node(self.dialog_manager.node)

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

            self._addstr(left_padding, 'Respond:\n', -1)
            for response in response_lines:
                for line in response:
                    self._addstr(left_padding, f'{line}\n', -1)
            self._addstr("", "\n", -1)

        self.dialog_pad.noutrefresh(0, 0)

    def add_message(self, character:core.Character, text:str) -> None:
        # message is left justified and indented to the right of the character
        # portrait

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
        while len(self.animation_queue) > 0 and self.animation_queue[0].animate(self.interface.gamestate.timestamp):
            self.animation_queue.popleft()

