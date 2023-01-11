""" Docked View """

from typing import Any
import curses
from curses import textpad

from stellarpunk import interface, core, config
from stellarpunk.interface import ui_utils

class StationView(interface.View):
    def __init__(self, station:core.Station, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.station = station

        # info pad sits to the left
        self.info_pad:interface.Canvas = None # type: ignore[assignment]

        # detail pad sits to the right
        self.detail_pad:interface.Canvas = None # type: ignore[assignment]

    def initialize(self) -> None:
        self.interface.reinitialize_screen(name="Station View")

        ipw = config.Settings.interface.StationView.info_width
        self.info_pad = interface.Canvas(
            curses.newpad(config.Settings.interface.StationView.info_lines, ipw),
            self.interface.viewscreen_height-2,
            ipw,
            self.interface.viewscreen_y+1,
            self.interface.viewscreen_x+1,
            self.interface.aspect_ratio(),
        )

        dpw = self.interface.viewscreen_width - ipw - 3
        self.detail_pad = interface.Canvas(
            curses.newpad(config.Settings.interface.StationView.detail_lines, dpw),
            self.interface.viewscreen_height-2,
            dpw,
            self.interface.viewscreen_y+1,
            self.info_pad.x+ipw+1,
            self.interface.aspect_ratio(),
        )
        self.detail_pad.window.scrollok(True)

        self.draw_station_data()
        self.scroll_info_pad(0)
        self.scroll_detail_pad(0)

    def scroll_info_pad(self, position:int) -> None:
        self.info_pad.noutrefresh(position, 0)

    def scroll_detail_pad(self, position:int) -> None:
        self.detail_pad.noutrefresh(position, 0)

    def draw_station_data(self) -> None:
        left_padding = int((self.info_pad.width - self.station.sprite.width)//2)-1
        textpad.rectangle(self.info_pad.window, 0, left_padding, self.station.sprite.height+1, left_padding+self.station.sprite.width+2)
        ui_utils.draw_sprite(self.station.sprite, self.info_pad, 1, left_padding+1)

        self.info_pad.window.move(self.station.sprite.height+3, 0)

        self.info_pad.window.addstr(f'{self.station.name}\n')
        self.info_pad.window.addstr(f'{self.station.address_str()}\n')
        self.info_pad.window.addstr("\n")

        self.info_pad.window.addstr("more info goes here")
        self.info_pad.window.addstr("\n")
