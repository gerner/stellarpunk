""" Docked View """

from typing import Any, Collection, List, Optional, Callable, Tuple
import curses
import curses.ascii
from curses import textpad
import textwrap
import enum
import time
import math

import numpy as np

from stellarpunk import interface, core, config, util, events
from stellarpunk.interface import ui_util, starfield
from stellarpunk.interface.ui_util import ValidationError


class CharacterMenuItem(ui_util.MenuItem):
    def __init__(self, character:core.Character, action:Callable[[], Any]) -> None:
        super().__init__()
        self.character = character
        self._action = action
        self._bbox = (0,0,0,0)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self._bbox

    @property
    def action(self) -> Callable[[], Any]:
        return self._action

    def draw(self, canvas: interface.BasicCanvas, y: int, x: int) -> None:
        attr = 0
        if self._selected:
            attr = curses.A_STANDOUT
        ui_util.draw_sprite(self.character.portrait, canvas, y, x+1)

        info_x = x + 1 + self.character.portrait.width + 1
        canvas.addstr(y, info_x, self.character.name, attr)
        canvas.addstr(y+1, info_x, self.character.short_id(), attr)

        info_width = max(len(self.character.name), len(self.character.short_id()))

        self._bbox = (y, x, y+self.character.portrait.height+1, x+self.character.portrait.width+2+info_width)


class Mode(enum.Enum):
    """ Station view UI modes, mutually exclusive things to display. """
    STATION_MENU = enum.auto()
    TRADE = enum.auto()
    PEOPLE = enum.auto()
    EXIT = enum.auto()


class StationView(interface.View):
    """ UI experience while docked at a station. """
    def __init__(
            self, station: core.Station, ship: core.Ship, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.station = station
        self.ship = ship

        # info pad sits to the left
        self.info_pad: interface.BasicCanvas = None  # type: ignore[assignment]

        # detail pad sits to the right
        self.detail_pad: interface.BasicCanvas = None  # type: ignore[assignment]

        self.detail_top_padding = 3
        self.detail_padding = 5

        # state for animated starfield background for station sprite
        self.station_sprite = self.station.sprite
        self.station_sprite_update_interval = 1.0
        self.station_sprite_stepsize = 0.25
        self.last_station_sprite_update = 0.

        self.mode = Mode.STATION_MENU
        self.station_menu = ui_util.Menu("uninitialized", [])
        self.sell_menu = ui_util.MeterMenu("uninitialized", [])
        self.buy_menu = ui_util.MeterMenu("uninitialized", [])
        self.people_menu = ui_util.Menu("uninitialized", [])

    def initialize(self) -> None:
        self.interface.reinitialize_screen(name="Station View")

        ipw = config.Settings.interface.StationView.info_width
        self.info_pad = self.interface.newpad(
            config.Settings.interface.StationView.info_lines, ipw,
            self.interface.viewscreen.height-2, ipw,
            self.interface.viewscreen.y+1, self.interface.viewscreen.x+1,
            self.interface.aspect_ratio,
        )

        dpw = self.interface.viewscreen.width - ipw - 3
        self.detail_pad = self.interface.newpad(
            config.Settings.interface.StationView.detail_lines, dpw,
            self.interface.viewscreen.height-2, dpw,
            self.interface.viewscreen.y+1, self.info_pad.x+ipw+1,
            self.interface.aspect_ratio,
        )

        self.enter_mode(self.mode)

    def focus(self) -> None:
        self.interface.reinitialize_screen(name="Station View")
        self.active=True

    def terminate(self) -> None:
        # transition to EXIT mode to leave, e.g., TRADE and relase pause lock
        # we might have already done this if we chose to exit docking
        self.enter_mode(Mode.EXIT)

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

    def enter_mode(self, mode:Mode) -> None:
        # leave the old mode (right now only trade has special leaving logic)
        if self.mode == Mode.TRADE:
            self._leave_trade()

        # enter the new mode
        self.mode = mode
        if mode == Mode.STATION_MENU:
            self._enter_station_menu()
        elif mode == Mode.TRADE:
            self._enter_trade()
        elif mode == Mode.PEOPLE:
            self._enter_people()
        elif mode == Mode.EXIT:
            # nothing to do on exit, should only happen if we're terminating
            self.logger.debug(f'exiting station view')
            pass
        else:
            raise ValueError(f'unknown mode {self.mode}')

    def _update_station_sprite(self) -> bool:
        now = time.perf_counter()
        if now - self.last_station_sprite_update < self.station_sprite_update_interval:
            return False

        self.last_station_sprite_update = now
        starfield_layers = self.gamestate.portrait_starfield
        perspective = interface.Perspective(
            interface.BasicCanvas(24*2, 48*2, 0, 0, 2.0),
            starfield_layers[0].zoom, starfield_layers[0].zoom, starfield_layers[1].zoom,
        )
        sf = starfield.Starfield(starfield_layers, perspective)
        dist = starfield_layers[0].zoom*self.station_sprite_stepsize*now
        perspective.cursor = util.polar_to_cartesian(dist, -math.pi/4.)

        starfield_sprite = sf.draw_starfield_to_sprite(self.station.sprite.width, self.station.sprite.height)
        self.station_sprite = ui_util.composite_sprites([starfield_sprite, self.station.sprite])
        return True

    def _draw_station_info(self) -> None:
        """ Draws overall station information in the info pad. """
        left_padding = int(
            (self.info_pad.width - self.station.sprite.width)//2
        )-1
        self.info_pad.rectangle(
            0,
            left_padding,
            self.station.sprite.height+1,
            left_padding+self.station.sprite.width+2
        )
        # TODO: scrolling star background
        if self._update_station_sprite():
            ui_util.draw_sprite(
                self.station_sprite, self.info_pad, 1, left_padding+1
            )

        y = self.station.sprite.height+3
        x = 0
        self.info_pad.addstr(y, x, f'{self.station.name}')
        self.info_pad.addstr(y+1, x, f'{self.station.address_str()}')

        self.info_pad.addstr(y+3, x, "more info goes here")
        self.info_pad.noutrefresh(0, 0)

    def _enter_station_menu(self) -> None:
        self.detail_pad.erase()
        self.mode = Mode.STATION_MENU
        self.station_menu = ui_util.Menu(
            "Station Menu",
            ui_util.number_text_menu_items([
                ui_util.TextMenuItem(
                    "Trade", lambda: self.enter_mode(Mode.TRADE)
                ),
                ui_util.TextMenuItem(
                    "People", lambda: self.enter_mode(Mode.PEOPLE)
                ),
                ui_util.TextMenuItem(
                    "Undock", lambda: self.interface.close_view(self)
                ),
            ])
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
        self.station_menu.draw(self.detail_pad, y, x)

        self.detail_pad.noutrefresh(0, 0)

    def _key_list_station_menu(self) -> Collection[interface.KeyBinding]:
        return self.station_menu.key_list()

    def _enter_trade(self) -> None:
        self.detail_pad.erase()
        self.gamestate.force_pause(self)
        self.mode = Mode.TRADE

        station_agent = self.gamestate.econ_agents[self.station.entity_id]
        pchain = self.gamestate.production_chain

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
            total_width=self.detail_pad.width-2*self.detail_padding,
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

    def _leave_trade(self) -> None:
        self.gamestate.force_unpause(self)

    def _validate_trade(self, *args:Any) -> bool:
        # validate the overall set of trades are valid
        # validate ship and station have the relevant goods
        # validate ship and station have total cargo capacity for buys/sells
        # validate station budgets for each resource for sells
        # validate player and station agent can afford all buys/sells

        # the trades will be conducted item-wise in any order, so they must all
        # be valid if conducted in any order
        # so we'll be as strict as possible:
        # the station must have capacity and budget for all player sales
        # the player must have capacity and budget for all player buys

        try:
            station_agent = self.gamestate.econ_agents[self.station.entity_id]
            ship_capacity = self.ship.cargo_capacity - np.sum(self.ship.cargo)
            station_capacity = self.station.cargo_capacity - np.sum(self.station.cargo)
            total_buy_amount = 0.
            total_sell_amount = 0.
            total_buy_value = 0.
            total_sell_value = 0.
            for item in self.sell_menu.options:
                resource = item.data
                assert isinstance(resource, int)
                resource_delta = item.setting - item.value
                # player sells case
                if resource_delta > 0:
                    raise ValidationError("The station does not sell that good")
                elif resource_delta < 0:
                    if self.ship.cargo[resource] + resource_delta < 0:
                        raise ValidationError("Your ship doesn't have any more of that good to sell")
                    trade_value = resource_delta * station_agent.buy_price(resource)
                    if station_agent.budget(resource) + trade_value < 0:
                        raise ValidationError("The station won't buy that many of those goods")
                    total_sell_amount -= resource_delta
                    total_sell_value -= trade_value

            for item in self.buy_menu.options:
                resource = item.data
                assert isinstance(resource, int)
                resource_delta = item.setting - item.value
                # player buys case
                if resource_delta < 0:
                    raise ValidationError("The station doen't buy that good")
                elif resource_delta > 0:
                    if station_agent.inventory(resource) - resource_delta < 0:
                        raise ValidationError("The station doesn't have any more of that good to sell")
                    total_buy_amount += resource_delta
                    total_buy_value += resource_delta * station_agent.sell_price(resource)

            if ship_capacity - total_buy_amount < 0:
                raise ValidationError("Your ship doesn't have the capacity to buy those goods")
            elif self.interface.player.character.balance - total_buy_value < 0:
                raise ValidationError("You don't have enough money to buy those goods")
            elif station_capacity - total_sell_amount < 0:
                raise ValidationError("The station doesn't have the capacity to buy those goods")
            elif station_agent.balance() - total_sell_value < 0:
                raise ValidationError("The station doesn't have enough money to buy those goods")

        except ValidationError as e:
            self.interface.status_message(
                e.message,
                self.interface.get_color(interface.Color.ERROR)
            )
            return False
        else:
            return True

    def _compute_trade_value(self) -> float:
        station_agent = self.gamestate.econ_agents[self.station.entity_id]
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
        self.sell_menu.draw(self.detail_pad, 2, self.detail_padding)
        self.buy_menu.draw(self.detail_pad, 3 + self.sell_menu.height, self.detail_padding)

        y = 4 + self.sell_menu.height + self.buy_menu.height
        trade_value = self._compute_trade_value()
        if trade_value > 0.:
            self.detail_pad.addstr(y, self.detail_padding, f'Trade Value: ${trade_value:.2f}')
        self.detail_pad.addstr(y+1, self.detail_padding, "Press <ENTER> to accept or <ESC> to cancel")
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
                self.enter_mode(Mode.STATION_MENU)

        def accept() -> None:
            station_agent = self.gamestate.econ_agents[self.station.entity_id]

            total_sell_value = 0.
            total_buy_value = 0.

            # conduct the sells player -> station
            for sell_item in self.sell_menu.options:
                if sell_item.setting == sell_item.value:
                    continue
                assert sell_item.setting < sell_item.value
                assert isinstance(sell_item.data, int)
                resource = sell_item.data
                price = station_agent.buy_price(resource)
                amount = sell_item.value - sell_item.setting
                self.gamestate.transact(
                    resource,
                    station_agent,
                    self.interface.player.agent,
                    price,
                    amount
                )
                total_sell_value += price * amount

            # conduct the buys station -> player
            for buy_item in self.buy_menu.options:
                if buy_item.setting == buy_item.value:
                    continue
                assert buy_item.setting > buy_item.value
                assert isinstance(buy_item.data, int)
                resource = buy_item.data
                price = station_agent.sell_price(resource)
                amount = buy_item.setting - buy_item.value
                self.gamestate.transact(
                    resource,
                    self.interface.player.agent,
                    station_agent,
                    price,
                    amount
                )
                total_buy_value += price*amount

            if total_sell_value > 0. or total_buy_value > 0.:
                self.interface.status_message(f'Trade completed: sold ${total_sell_value:.2f} bought ${total_buy_value:.2f}, total: ${total_sell_value - total_buy_value:.2f}')
            else:
                self.interface.status_message("No trade conducted")

            # return to station menu
            self.enter_mode(Mode.STATION_MENU)

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
        self.detail_pad.erase()
        self.mode = Mode.PEOPLE

        def make_contact(character: core.Character) -> Callable[[], Any]:
            def handle_contact(c: core.Character) -> None:
                self.gamestate.trigger_event_immediate(
                    [character],
                    events.e(events.Events.CONTACT),
                    {
                        events.ck(events.ContextKeys.CONTACTER): self.interface.player.character.short_id_int(),
                    },
                )

            def contact() -> None:
                self.logger.debug(f'contacting {character.short_id()}')
                number_str = "".join(list(f'{oct(x)[2:]:0>4}' for x in character.entity_id.bytes[0:8]))
                s = number_str
                self.interface.log_message(f'dialing {s[0:8]}-{s[8:16]}-{s[16:24]}-{s[24:32]}...')
                self.interface.mixer.play_sample(
                    ui_util.dtmf_sample(number_str, self.interface.mixer.sample_rate),
                    lambda: handle_contact(character)
                )
            return contact
        self.people_menu = ui_util.Menu(
            "Station Directory",
            [
                CharacterMenuItem(x, make_contact(x))
                for x in self.gamestate.characters_by_location[self.station.entity_id]
            ],
            -1
        )

    def _draw_people(self) -> None:
        y = 1
        x = self.detail_padding
        self.people_menu.draw(self.detail_pad, y, x)
        y += self.people_menu.height+1
        self.detail_pad.addstr(y, x, "Press <ESC> to cancel")
        self.detail_pad.noutrefresh(0, 0)

    def _key_list_people(self) -> Collection[interface.KeyBinding]:
        for character in self.gamestate.characters_by_location[self.station.entity_id]:
            pass

        def contact() -> None:
            self.gamestate.trigger_event_immediate(
                [character],
                events.e(events.Events.CONTACT),
                {
                    events.ck(events.ContextKeys.CONTACTER): self.interface.player.character.short_id_int(),
                },
            )
        key_list:List[interface.KeyBinding] = []
        key_list.extend(self.people_menu.key_list())
        key_list.extend({
            self.bind_key(curses.ascii.ESC, lambda: self.enter_mode(Mode.STATION_MENU), help_key="station_people_cancel"),
        })
        return key_list
