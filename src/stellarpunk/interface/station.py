""" Docked View """

from typing import Any, Collection, List
import curses
import curses.ascii
from curses import textpad
import textwrap
import enum
import time
import math

import numpy as np

from stellarpunk import interface, core, config, util
from stellarpunk.interface import ui_util, starfield


class Mode(enum.Enum):
    """ Station view UI modes, mutually exclusive things to display. """
    NONE = enum.auto()
    STATION_MENU = enum.auto()
    TRADE = enum.auto()
    PEOPLE = enum.auto()


class StationView(interface.View):
    """ UI experience while docked at a station. """
    def __init__(
            self, station: core.Station, ship: core.Ship, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.station = station
        self.ship = ship

        # info pad sits to the left
        self.info_pad: interface.Canvas = None  # type: ignore[assignment]

        # detail pad sits to the right
        self.detail_pad: interface.Canvas = None  # type: ignore[assignment]

        self.detail_top_padding = 3
        self.detail_padding = 5

        # state for animated starfield background for station sprite
        self.station_sprite = self.station.sprite
        self.station_sprite_update_interval = 1.0
        self.station_sprite_stepsize = 0.25
        self.last_station_sprite_update = 0.

        self.mode = Mode.NONE
        self.station_menu = ui_util.Menu("uninitialized", [])
        self.sell_menu = ui_util.MeterMenu("uninitialized", [])
        self.buy_menu = ui_util.MeterMenu("uninitialized", [])
        self._enter_station_menu()

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

    def update_display(self) -> None:
        self._draw_station_info()
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

    def _update_station_sprite(self) -> None:
        now = time.perf_counter()
        if now - self.last_station_sprite_update < self.station_sprite_update_interval:
            return
        self.logger.info(f'updating sprite')
        self.last_station_sprite_update = now
        starfield_layers = self.interface.gamestate.portrait_starfield
        perspective = interface.Perspective(
            interface.BasicCanvas(24*2, 48*2, 0, 0, 2.0),
            starfield_layers[0].zoom, starfield_layers[0].zoom, starfield_layers[1].zoom,
        )
        sf = starfield.Starfield(starfield_layers, perspective)
        dist = starfield_layers[0].zoom*self.station_sprite_stepsize*now
        perspective.cursor = util.polar_to_cartesian(dist, -math.pi/4.)

        starfield_sprite = sf.draw_starfield_to_sprite(self.station.sprite.width, self.station.sprite.height)
        self.station_sprite = core.Sprite.composite_sprites([starfield_sprite, self.station.sprite])

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
        # TODO: scrolling star background
        self._update_station_sprite()
        ui_util.draw_sprite(
            self.station_sprite, self.info_pad, 1, left_padding+1
        )
        #ui_util.draw_sprite(
        #    self.station.sprite, self.info_pad, 1, left_padding+1
        #)

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
                    "Trade", self._enter_trade
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

        self.detail_pad.erase()

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
        return self.station_menu.key_list()

    def _enter_trade(self) -> None:
        self.interface.gamestate.force_pause(self)
        self.mode = Mode.TRADE

        station_agent = self.interface.gamestate.econ_agents[self.station.entity_id]
        pchain = self.interface.gamestate.production_chain

        # stuff we can sell that station buys
        sell_items:List[ui_util.MeterItem] = []
        for resource in station_agent.buy_resources():
            price_str = f'${station_agent.buy_price(resource):.2f}'
            sell_items.append(ui_util.MeterItem(
                f'{price_str:>8} {pchain.product_names[resource]}',
                int(self.ship.cargo[resource]),
                maximum=int(self.ship.cargo_capacity),
                pool=int(station_agent.inventory(resource)),
                data=int(resource),
            ))

        self.sell_menu = ui_util.MeterMenu(
            "Sell to Station",
            sell_items,
            validator=self._validate_trade,
        )

        # stuff we can buy that station sells
        buy_items:List[ui_util.MeterItem] = []
        for resource in station_agent.sell_resources():
            price_str = f'${station_agent.sell_price(resource):.2f}'
            buy_items.append(ui_util.MeterItem(
                f'{price_str:>8} {pchain.product_names[resource]}',
                int(self.ship.cargo[resource]),
                maximum=int(self.ship.cargo_capacity),
                pool=int(station_agent.inventory(resource)),
                data=int(resource),
            ))

        self.buy_menu = ui_util.MeterMenu(
            "Buy from Station",
            buy_items,
            validator=self._validate_trade,
        )
        self.buy_menu.selected_option = -1

    def _validate_trade(self, *args:Any) -> bool:
        # validate the overall set of trades are valid
        # validate ship and station have the relevant goods
        # validate ship and station have total cargo capacity
        # validate station budgets for each resource
        # validate player and station agent can afford the overall trade

        station_agent = self.interface.gamestate.econ_agents[self.station.entity_id]
        ship_capacity = self.ship.cargo_capacity - np.sum(self.ship.cargo)
        station_capacity = self.station.cargo_capacity - np.sum(self.station.cargo)
        total_trade_value = 0.
        for item in self.sell_menu.options + self.buy_menu.options:
            resource = item.data
            assert isinstance(resource, int)
            resource_delta = item.setting - item.value
            if resource_delta > 0:
                if station_agent.inventory(resource) - resource_delta < 0:
                    self.interface.status_message("The station doesn't have any more of that good to sell", self.interface.error_color)
                    return False
                trade_value = resource_delta * station_agent.sell_price(resource)
            else:
                if self.ship.cargo[resource] + resource_delta < 0:
                    self.interface.status_message("Your ship doesn't have any more of that good to sell", self.interface.error_color)
                    return False
                trade_value = resource_delta * station_agent.buy_price(resource)
                if station_agent.budget(resource) + trade_value < 0:
                    self.interface.status_message("The station won't buy that many of those goods", self.interface.error_color)
                    return False

            ship_capacity -= resource_delta
            station_capacity += resource_delta
            total_trade_value += trade_value

        if ship_capacity < 0:
            self.interface.status_message("Your ship doesn't have the capacity for those goods", self.interface.error_color)
            return False
        if self.interface.gamestate.player.character.balance - total_trade_value < 0:
            self.interface.status_message("You don't have enough money to buy those goods", self.interface.error_color)
            return False

        if station_capacity < 0:
            self.interface.status_message("The station doesn't have the capacity for those goods", self.interface.error_color)
            return False
        if station_agent.balance() + total_trade_value < 0:
            self.interface.status_message("The station doesn't have enough money to buy all those goods", self.interface.error_color)
            return False

        return True

    def _compute_trade_value(self) -> float:
        station_agent = self.interface.gamestate.econ_agents[self.station.entity_id]
        total_trade_value = 0.
        for item in self.sell_menu.options + self.buy_menu.options:
            resource = item.data
            assert isinstance(resource, int)
            resource_delta = item.setting - item.value
            if resource_delta > 0:
                trade_value = resource_delta * station_agent.sell_price(resource)
            else:
                trade_value = resource_delta * station_agent.buy_price(resource)
            total_trade_value += trade_value
        return total_trade_value

    def _draw_trade(self) -> None:
        self.detail_pad.erase()
        self.sell_menu.draw(self.detail_pad, 2, 1)
        self.buy_menu.draw(self.detail_pad, 3 + self.sell_menu.height, 1)

        y = 4 + self.sell_menu.height + self.buy_menu.height
        trade_value = self._compute_trade_value()
        if trade_value > 0.:
            self.detail_pad.addstr(y, 1, f'Trade Value: ${trade_value:.2f}')
        self.detail_pad.addstr(y+1, 1, "Press <ENTER> to accept or <ESC> to cancel")
        self.detail_pad.noutrefresh(0, 0)

    def _key_list_trade(self) -> Collection[interface.KeyBinding]:
        if self.sell_menu.selected_option >= 0:
            prev_menu = self.buy_menu
            current_menu = self.sell_menu
            next_menu = self.buy_menu
        else:
            prev_menu = self.sell_menu
            current_menu = self.buy_menu
            next_menu = self.sell_menu

        def sel_next() -> None:
            if current_menu.selected_option == len(current_menu.options)-1:
                current_menu.selected_option = -1
                next_menu.selected_option = 0
            else:
                current_menu.select_next()

        def sel_prev() -> None:
            if current_menu.selected_option == 0:
                current_menu.selected_option = -1
                prev_menu.selected_option = len(prev_menu.options)-1
            else:
                current_menu.select_prev()

        def cancel() -> None:
            any_different = any(map(lambda x: x.setting != x.value, self.buy_menu.options + self.sell_menu.options))
            if any_different:
                self._enter_trade()
            else:
                self.interface.gamestate.force_unpause(self)
                self._enter_station_menu()

        def accept() -> None:
            # conduct the trade
            #self.interface.gamestate.force_unpause(self)
            pass

        key_list:List[interface.KeyBinding] = []
        key_list.extend(self.bind_aliases(
            [ord("j"), ord("s"), curses.KEY_DOWN],
            sel_next, help_key="station_trade_nav"
        ))
        key_list.extend(self.bind_aliases(
            [ord("k"), ord("w"), curses.KEY_UP],
            sel_prev, help_key="station_trade_nav"
        ))
        key_list.extend(self.bind_aliases(
            [ord("l"), ord("d"), curses.KEY_RIGHT],
            current_menu.select_more, help_key="station_trade_inc"
        ))
        key_list.extend(self.bind_aliases(
            [ord("h"), ord("a"), curses.KEY_LEFT],
            current_menu.select_less, help_key="station_trade_inc"
        ))
        key_list.extend(self.bind_aliases(
            [ord("L"), ord("D"), curses.KEY_SRIGHT],
            lambda: current_menu.select_more(increment=10),
            help_key="station_trade_biginc"
        ))
        key_list.extend(self.bind_aliases(
            [ord("H"), ord("A"), curses.KEY_SLEFT],
            lambda: current_menu.select_less(increment=10),
            help_key="station_trade_biginc"
        ))
        key_list.extend([
            self.bind_key(curses.ascii.ESC, cancel, help_key="station_trade_cancel"),
            self.bind_key(curses.ascii.CR, accept, help_key="station_trade_accept"),
        ])
        return key_list

    def _enter_people(self) -> None:
        pass

    def _draw_people(self) -> None:
        pass

    def _key_list_people(self) -> Collection[interface.KeyBinding]:
        return []
