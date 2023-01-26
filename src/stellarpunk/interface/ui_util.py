""" Utils for drawing Stellarpunk UI """

import curses
import math
from typing import List, Callable, Any, Collection, Optional, Sequence, Dict, Tuple
from dataclasses import dataclass

import drawille # type: ignore

from stellarpunk import core, interface, config, util


def composite_sprites(sprites:Sequence[core.Sprite]) -> core.Sprite:
    if len(sprites) == 0:
        raise ValueError("no sprites to composite")

    text = [[" "]*sprites[0].width for _ in range(sprites[0].height)]
    attr:Dict[Tuple[int,int], Tuple[int, int]] = {}

    for sprite in sprites:
        for y, row in enumerate(sprite.text):
            for x, c in enumerate(sprite.text[y]):
                if c != " " and c != chr(drawille.braille_char_offset):
                    text[y][x] = c
                    if (x,y) in sprite.attr:
                        attr[x,y] = sprite.attr[x,y]
                    else:
                        attr[x,y] = (0,0)

    return core.Sprite(["".join(t) for t in text], attr)

def draw_sprite(
        sprite: core.Sprite, canvas: interface.BasicCanvas, y: int, x: int) -> None:
    y_off = 0
    for row in sprite.text:
        for x_off, s in enumerate(row):
            if (x_off, y_off) in sprite.attr:
                a, c = sprite.attr[x_off, y_off]
                canvas.addstr(
                    y+y_off, x+x_off,
                    s,
                    a | curses.color_pair(c)
                )
            else:
                canvas.addstr(y+y_off, x+x_off, s)
        y_off += 1


def draw_portrait(portrait: core.Sprite, canvas: interface.Canvas) -> None:
    padding = " " * ((canvas.width-portrait.width)//2)
    for row in portrait.text:
        canvas.window.addstr(padding+f'{row}\n')


def draw_heading(canvas: interface.Canvas, text: str) -> None:
    canvas.window.addstr("\t")
    canvas.window.addstr(f' {text} ', curses.A_UNDERLINE | curses.A_BOLD)
    canvas.window.addstr("\n\n")


def product_name(
        production_chain: core.ProductionChain,
        product_id: Optional[int],
        max_length: int=1024
) -> str:
    if product_id is None:
        return util.elipsis("None", max_length)
    else:
        return util.elipsis(production_chain.product_names[product_id], max_length)


class ValidationError(Exception):
    def __init__(self, message:str):
        super().__init__(message)
        self.message = message


@dataclass
class MenuItem:
    label: str
    action: Callable[[], Any]


class Menu:
    def __init__(
            self,
            title: str,
            options: List[MenuItem],
            selected_option: int = 0
    ) -> None:
        self.title = title
        self.options = options
        self.selected_option = selected_option

    def select_option(self, selected_option: int) -> None:
        self.selected_option = selected_option

    def select_next(self) -> int:
        if self.selected_option < len(self.options)-1:
            self.selected_option += 1
        return self.selected_option

    def select_prev(self) -> int:
        if self.selected_option > 0:
            self.selected_option -= 1
        return self.selected_option

    def activate_item(self) -> Any:
        self.options[self.selected_option].action()

    def draw_menu(self, canvas: interface.BasicCanvas, y: int, x: int) -> None:
        canvas.addstr(
            y, x,
            f' {self.title} \n\n',
            curses.A_UNDERLINE | curses.A_BOLD
        )

        y += 2
        for i, option in enumerate(self.options):
            attr = 0
            if i == self.selected_option:
                attr |= curses.A_STANDOUT
            canvas.addstr(
                y, x,
                f'{i+1:>{len(self.options)//10+1}}. {option.label}\n',
                attr
            )
            y += 1

    def key_list(self) -> Collection[interface.KeyBinding]:
        nav_help = config.key_help(self, "j")
        act_help = config.key_help(self, "\r")
        return [
            interface.KeyBinding(ord("j"), self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("k"), self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("w"), self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("s"), self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(curses.KEY_DOWN, self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(curses.KEY_UP, self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("\r"), self.activate_item, act_help, help_key="menu_act"),
        ]


class MeterItem:
    def __init__(
            self,
            label: str,
            value: int,
            setting: Optional[int] = None,
            minimum: int = 0,
            maximum: int = 100,
            increment: int = 1,
            pool: Optional[int] = None,
            data: Optional[Any] = None,
    ) -> None:
        self.label = label
        self.value = value
        if setting is None:
            setting = self.value
        self.setting = setting
        self.minimum = minimum
        self.maximum = maximum
        self.increment = increment
        self.pool = pool
        self.data = data


class MeterMenu:
    @staticmethod
    def validate_pool(item: MeterItem, meter: "MeterMenu") -> bool:
        # no pool or amount above value is less than the pool
        return item.pool is None or item.pool - (item.setting - item.value) >= 0

    def __init__(
            self,
            title: str,
            options: List[MeterItem],
            total_width: int = 128,
            label_width: int = 32,
            number_width: int = 7,
            validator: Optional[Callable[[MeterItem, "MeterMenu"], bool]] = None,
    ) -> None:
        self.title = title
        self.options = options
        self.selected_option = 0
        if validator is None:
            validator = MeterMenu.validate_pool
        self.validator = validator

        self.total_width = total_width
        self.label_width = label_width
        self.left_number_width = 2*number_width+1
        self.right_number_width = number_width
        self.meter_width = total_width - (
            label_width + 1 +
            self.left_number_width + 1 +
            self.right_number_width + 1
        )

    @property
    def width(self) -> int:
        return self.total_width

    @property
    def height(self) -> int:
        # title, blank line, all the meters
        return 2 + len(self.options)

    def select_option(self, selected_option: int) -> None:
        self.selected_option = selected_option

    def select_next(self) -> int:
        if self.selected_option < len(self.options)-1:
            self.selected_option += 1
        return self.selected_option

    def select_prev(self) -> int:
        if self.selected_option > 0:
            self.selected_option -= 1
        return self.selected_option

    def select_more(self, increment: Optional[int] = None) -> int:
        option = self.options[self.selected_option]
        if increment is None:
            increment = option.increment
        old_setting = option.setting
        option.setting += min(increment, option.maximum-option.setting)
        if not self.validator(option, self):
            option.setting = old_setting
        return option.setting

    def select_less(self, increment: Optional[int] = None) -> int:
        option = self.options[self.selected_option]
        if increment is None:
            increment = option.increment
        old_setting = option.setting
        option.setting -= min(increment, option.setting-option.minimum)
        if not self.validator(option, self):
            option.setting = old_setting
        return option.setting

    def draw(self, canvas: interface.BasicCanvas, y: int, x: int) -> None:
        canvas.addstr(
            y, x,
            f' {self.title} \n\n',
            curses.A_UNDERLINE | curses.A_BOLD
        )

        y += 2
        for i, option in enumerate(self.options):
            attr = 0
            if i == self.selected_option:
                attr |= curses.A_STANDOUT
            canvas.addstr(
                y, x,
                f'{option.label:>{self.label_width}} ',
                attr
            )

            self._draw_meter(canvas, option, y, x+self.label_width+1)

            y += 1

    def key_list(self) -> Collection[interface.KeyBinding]:
        nav_help = config.key_help(self, "j")
        inc_help = config.key_help(self, "d")
        act_help = config.key_help(self, "\r")
        return [
            interface.KeyBinding(ord("j"), self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("k"), self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("s"), self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("w"), self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(curses.KEY_DOWN, self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(curses.KEY_UP, self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("d"), self.select_more, inc_help, help_key="menu_inc"),
            interface.KeyBinding(ord("a"), self.select_less, inc_help, help_key="menu_inc"),
            interface.KeyBinding(ord("l"), self.select_more, inc_help, help_key="menu_inc"),
            interface.KeyBinding(ord("h"), self.select_less, inc_help, help_key="menu_inc"),
            interface.KeyBinding(curses.KEY_RIGHT, self.select_more, inc_help, help_key="menu_inc"),
            interface.KeyBinding(curses.KEY_LEFT, self.select_less, inc_help, help_key="menu_inc"),
        ]

    def _draw_meter(
            self,
            canvas: interface.BasicCanvas,
            option: MeterItem,
            y: int, x: int,
            attr: int = 0
    ) -> None:
        # current setting right justified
        if option.setting > option.value:
            diff_str = f'(+{option.setting-option.value}) '
        elif option.setting < option.value:
            diff_str = f'(-{option.value - option.setting}) '
        else:
            diff_str = ""

        value_str = f'{diff_str}{option.setting}'
        canvas.addstr(y, x, f'{value_str:>{self.left_number_width}} ', attr)

        # meter visual █░▓▁
        # show original value and current setting
        # shade the difference between value and current setting
        # any non-zero amount should be indicated
        chars_per_unit = self.meter_width / option.maximum

        left_width = int(math.ceil(option.value*chars_per_unit))
        l_filled_str = "█" * int(math.ceil(min(option.setting, option.value)*chars_per_unit))
        left_str = f'{l_filled_str:{"░"}<{left_width}}'

        right_width = self.meter_width - left_width
        r_filled_str = "▓" * int(math.ceil(option.setting - option.value)*chars_per_unit)
        right_str = f'{r_filled_str:{"▁"}<{right_width}}'

        meter_visual = left_str + right_str

        assert len(meter_visual) <= self.meter_width
        canvas.addstr(y, x + self.left_number_width + 1, meter_visual, attr)

        # meter maximum right justified
        if option.pool is not None:
            canvas.addstr(
                y, x + self.left_number_width + 1 + self.meter_width + 1,
                f' {option.pool - (option.setting-option.value):>{self.right_number_width}}'
            )
