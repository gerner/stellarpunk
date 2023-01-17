""" Character Information View """

from typing import Any
import curses

from stellarpunk import core, interface, config
from stellarpunk.interface import ui_util

class CharacterView(interface.View):
    def __init__(self, character:core.Character, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.character = character

        # info pad sits to the left
        self.info_pad:interface.Canvas = None # type: ignore[assignment]

        # detail pad sits to the right
        self.detail_pad:interface.Canvas = None # type: ignore[assignment]

    def initialize(self) -> None:
        self.interface.reinitialize_screen(name="Character Viewer")

        ipw = config.Settings.interface.CharacterView.info_width
        self.info_pad = interface.Canvas(
            curses.newpad(config.Settings.interface.CharacterView.info_lines, ipw),
            self.interface.viewscreen.height-2,
            ipw,
            self.interface.viewscreen.y+1,
            self.interface.viewscreen.x+1,
            self.interface.aspect_ratio,
        )

        dpw = self.interface.viewscreen.width - ipw - 3
        self.detail_pad = interface.Canvas(
            curses.newpad(config.Settings.interface.CharacterView.detail_lines, dpw),
            self.interface.viewscreen.height-2,
            dpw,
            self.interface.viewscreen.y+1,
            self.info_pad.x+ipw+1,
            self.interface.aspect_ratio,
        )
        self.detail_pad.window.scrollok(True)

        self.draw_character_data()
        self.scroll_info_pad(0)
        self.scroll_detail_pad(0)

    def scroll_info_pad(self, position:int) -> None:
        self.info_pad.noutrefresh(position, 0)

    def scroll_detail_pad(self, position:int) -> None:
        self.detail_pad.noutrefresh(position, 0)

    def draw_character_data(self) -> None:
        self.draw_info()
        self.draw_detail()

    def draw_info(self) -> None:
        ui_util.draw_portrait(self.character.portrait, self.info_pad)
        self.info_pad.window.addstr('\n')

        self.info_pad.window.addstr(f'{self.character.name}\n')
        self.info_pad.window.addstr(f'{self.character.location.address_str()}\n')
        self.info_pad.window.addstr("\n")

        self.info_pad.window.addstr("more info goes here")
        self.info_pad.window.addstr("\n")

    def draw_detail(self) -> None:
        ui_util.draw_heading(self.detail_pad, "Agenda")
        if len(self.character.agenda) == 0:
            self.detail_pad.window.addstr("no agenda\n")
        else:
            for agendum in self.character.agenda:
                self.detail_pad.window.addstr(f'* {agendum}\n')
        self.detail_pad.window.addstr("\n")

        ui_util.draw_heading(self.detail_pad, "Assets")
        if len(self.character.assets) == 0:
            self.detail_pad.window.addstr("no assets\n")
        else:
            for asset in self.character.assets:
                self.detail_pad.window.addstr(f'* {asset}\n')
        self.detail_pad.window.addstr("\n")

        ui_util.draw_heading(self.detail_pad, "History")
        self.detail_pad.window.addstr("\n")
