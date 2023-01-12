""" Docked View """

from typing import Any, Collection
import curses
from curses import textpad
import textwrap
import enum

from stellarpunk import interface, core, config
from stellarpunk.interface import ui_util


class Mode(enum.Enum):
    """ Station view UI modes, mutually exclusive things to display. """
    NONE = enum.auto()
    STATION_MENU = enum.auto()
    TRADE = enum.auto()
    PEOPLE = enum.auto()


class StationView(interface.View):
    """ UI experience while docked at a station. """
    def __init__(
            self, station: core.Station, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.station = station

        # info pad sits to the left
        self.info_pad: interface.Canvas = None  # type: ignore[assignment]

        # detail pad sits to the right
        self.detail_pad: interface.Canvas = None  # type: ignore[assignment]

        self.detail_top_padding = 3
        self.detail_padding = 5

        self.mode = Mode.NONE
        self.station_menu = ui_util.Menu("uninitialized", [])

    def initialize(self) -> None:
        self.interface.reinitialize_screen(name="Station View")

        ipw = config.Settings.interface.StationView.info_width
        self.info_pad = interface.Canvas(
            curses.newpad(
                config.Settings.interface.StationView.info_lines, ipw),
            self.interface.viewscreen_height-2,
            ipw,
            self.interface.viewscreen_y+1,
            self.interface.viewscreen_x+1,
            self.interface.aspect_ratio(),
        )

        dpw = self.interface.viewscreen_width - ipw - 3
        self.detail_pad = interface.Canvas(
            curses.newpad(
                config.Settings.interface.StationView.detail_lines, dpw),
            self.interface.viewscreen_height-2,
            dpw,
            self.interface.viewscreen_y+1,
            self.info_pad.x+ipw+1,
            self.interface.aspect_ratio(),
        )
        self.detail_pad.window.scrollok(True)

        self._draw_station_info()

        self._enter_station_menu()

    def update_display(self) -> None:
        if self.mode == Mode.STATION_MENU:
            self._draw_station_menu()
        elif self.mode == Mode.TRADE:
            self._draw_trade()
        elif self.mode == Mode.PEOPLE:
            self._draw_people()
        else:
            raise ValueError(f'unknown mode {self.mode}')

    def key_list(self) -> Collection[interface.KeyBinding]:
        if self.mode == Mode.STATION_MENU:
            return self._key_list_station_menu()
        elif self.mode == Mode.TRADE:
            return self._key_list_trade()
        elif self.mode == Mode.PEOPLE:
            return self._key_list_people()
        else:
            raise ValueError(f'unknown mode {self.mode}')

    def _draw_station_info(self) -> None:
        """ Draws overall station information in the info pad. """
        left_padding = int(
            (self.info_pad.width - self.station.sprite.width)//2
        )-1
        textpad.rectangle(
            self.info_pad.window,
            0,
            left_padding,
            self.station.sprite.height+1,
            left_padding+self.station.sprite.width+2
        )
        ui_util.draw_sprite(
            self.station.sprite, self.info_pad, 1, left_padding+1
        )

        self.info_pad.window.move(self.station.sprite.height+3, 0)

        self.info_pad.window.addstr(f'{self.station.name}\n')
        self.info_pad.window.addstr(f'{self.station.address_str()}\n')
        self.info_pad.window.addstr("\n")

        self.info_pad.window.addstr("more info goes here")
        self.info_pad.window.addstr("\n")
        self.info_pad.noutrefresh(0, 0)

    def _enter_station_menu(self) -> None:
        self.mode = Mode.STATION_MENU
        self.station_menu = ui_util.Menu(
            "Station Menu",
            [
                ui_util.MenuItem(
                    "Option A", lambda: self.interface.log_message("Option A")
                ),
                ui_util.MenuItem(
                    "Option B", lambda: self.interface.log_message("Option B")
                ),
                ui_util.MenuItem(
                    "Option C", lambda: self.interface.log_message("Option C")
                ),
                ui_util.MenuItem(
                    "Option D", lambda: self.interface.log_message("Option D")
                ),
            ]
        )

    def _draw_station_menu(self) -> None:
        """ Draws the main station menu of options. """

        description_lines = textwrap.wrap(
            self.station.description,
            width=self.detail_pad.width - 2*self.detail_padding
        )
        y = self.detail_top_padding
        x = self.detail_padding

        for line in description_lines:
            self.detail_pad.addstr(y, x, line)
            y += 1

        y += 1
        self.station_menu.draw_menu(self.detail_pad, y, x)

        self.detail_pad.noutrefresh(0, 0)

    def _key_list_station_menu(self) -> Collection[interface.KeyBinding]:
        return [
            self.bind_key(ord('j'), self.station_menu.select_next, help_key="station_menu_nav"),
            self.bind_key(ord('k'), self.station_menu.select_prev, help_key="station_menu_nav"),
            self.bind_key(ord('s'), self.station_menu.select_next, help_key="station_menu_nav"),
            self.bind_key(ord('w'), self.station_menu.select_prev, help_key="station_menu_nav"),
            self.bind_key(ord('\r'), self.station_menu.activate_item),
        ]

    def _enter_trade(self) -> None:
        pass

    def _draw_trade(self) -> None:
        pass

    def _key_list_trade(self) -> Collection[interface.KeyBinding]:
        return []

    def _enter_people(self) -> None:
        pass

    def _draw_people(self) -> None:
        pass

    def _key_list_people(self) -> Collection[interface.KeyBinding]:
        return []
