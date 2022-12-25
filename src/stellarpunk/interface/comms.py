""" Comms view between a character and the player """

from typing import Any, Optional, Sequence, Collection
import curses
import textwrap

from stellarpunk import core, interface, config, dialog
from stellarpunk.interface import ui_utils

class CommsView(interface.View):
    def __init__(self, dialog_graph:dialog.DialogGraph, *args:Any, **kwargs:Any) -> None:
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

        self.last_character:Optional[core.Character] = None

        self.dialog_graph = dialog_graph
        self.current_node:dialog.DialogNode = dialog_graph.nodes[dialog_graph.root_id]

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

        self.handle_dialog_node(self.dialog_graph.nodes[self.dialog_graph.root_id])

    def handle_dialog_node(self, node:dialog.DialogNode) -> None:
        self.current_node = node
        character = self.interface.gamestate.player.character
        self.add_message(character, node.text)
        self.add_responses(node)

    def add_responses(self, node:dialog.DialogNode) -> None:
        responses = node.choices
        left_padding = " " * self.response_indent

        if len(responses) == 0:
            assert node.terminal
            self.dialog_pad.window.addstr(f'{left_padding}press <ENTER> to close\n\n')
        elif len(responses) == 1 and responses[0].text == "":
            self.dialog_pad.window.addstr(f'{left_padding}press <ENTER> to continue\n\n')
        else:
            # indented to the right
            response_lines = [
                textwrap.wrap(
                    f' {i}: {response.text}',
                    width=self.response_width,
                    subsequent_indent=" "*4,
                ) for i, response in enumerate(responses, start=1)
            ]

            self.dialog_pad.window.addstr(f'{left_padding}Respond:\n')
            for response in response_lines:
                for line in response:
                    self.dialog_pad.window.addstr(f'{left_padding}{line}\n')
            self.dialog_pad.window.addstr("\n")

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

        for (p,d) in zip(portrait_lines, dialog_lines):
            self.dialog_pad.window.addstr(portrait_padding_l+p+portrait_padding_r+d)
            self.dialog_pad.window.addstr("\n")

        self.dialog_pad.window.addstr("\n")
        self.dialog_pad.noutrefresh(0, 0)

    def add_player_message(self, text:str) -> None:
        dialog_lines = textwrap.wrap(text, width=self.dialog_width)

        for d in dialog_lines:
            self.dialog_pad.window.addstr(f'{d:>{self.dialog_width+self.info_width+self.padding}}\n')

        self.dialog_pad.window.addstr("\n")
        self.dialog_pad.noutrefresh(0, 0)

    def choose_dialog_option(self, i:int) -> None:
        self.logger.info(f'pressed {i} corresponding to "{self.current_node.choices[i].text}" -> {self.current_node.choices[i].node_id}')
        chosen_option = self.current_node.choices[i]
        self.add_player_message(chosen_option.text)
        self.handle_dialog_node(self.dialog_graph.nodes[chosen_option.node_id])

    def bind_dialog_option_key(self, key:int, i:int) -> interface.KeyBinding:
        return self.bind_key(key, lambda: self.choose_dialog_option(i))

    def key_list(self) -> Collection[interface.KeyBinding]:
        keys = []
        if self.current_node.terminal:
            keys.append(self.bind_key(ord("\r"), lambda: self.interface.close_view(self)))
        elif len(self.current_node.choices) == 1 and self.current_node.choices[0].text == "":
            keys.append(self.bind_dialog_option_key(ord("\r"), 0))
        else:
            keys.extend([
                    self.bind_dialog_option_key(ord(str(i+1)), i)
                    for i in range(len(self.current_node.choices))
            ])
        return keys
