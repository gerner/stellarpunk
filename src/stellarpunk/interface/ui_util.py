""" Utils for drawing Stellarpunk UI """

import abc
import logging
import curses
import math
from typing import List, Callable, Any, Collection, Optional, Sequence, Dict, Tuple, Union
from dataclasses import dataclass

import drawille # type: ignore
import dtmf # type: ignore
import numpy as np
import numpy.typing as npt

from stellarpunk import core, interface, config, util

def initialize() -> None:
    dtmf._info._freq_map["dial"] = [350.0, 440.0]
    dtmf._info._freq_map["busy"] = [480.0, 620.0]
    dtmf._info._freq_map["ringing"] = [440.0, 480.0]
    dtmf._info._freq_map["zip"] = [440.0]

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

    return core.Sprite("ephemeral_composited_sprite", ["".join(t) for t in text], attr)

def draw_sprite(
    sprite: core.Sprite,
    canvas: interface.BasicCanvas,
    y: int,
    x: int,
    attr: int = 0,
) -> None:
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
                canvas.addstr(y+y_off, x+x_off, s, attr)
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


class UIComponent(abc.ABC):
    @property
    @abc.abstractmethod
    def bbox(self) -> Tuple[int, int, int, int]: ...

    @property
    def width(self) -> int:
        return self.bbox[3]-self.bbox[1]

    @property
    def height(self) -> int:
        return self.bbox[2]-self.bbox[0]

    @abc.abstractmethod
    def draw(self, canvas: interface.BasicCanvas, y: int, x: int) -> None: ...


class ValidationError(Exception):
    def __init__(self, message:str):
        super().__init__(message)
        self.message = message


class MenuItem(UIComponent, abc.ABC):
    def __init__(self) -> None:
        self._selected = False

    @property
    @abc.abstractmethod
    def action(self) -> Callable[[], Any]: ...

    def select(self) -> None:
        self._selected = True
    def deselect(self) -> None:
        self._selected = False


class TextMenuItem(MenuItem):
    def __init__(self, label:str, action: Callable[[], Any]) -> None:
        super().__init__()
        self.label = label
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
            attr |= curses.A_STANDOUT
        canvas.addstr(
            y, x,
            self.label,
            attr
        )
        self._bbox = (y, x, y+1, x+len(self.label))


def number_text_menu_items(options: List[TextMenuItem]) -> List[TextMenuItem]:
    number_width = len(options)//10+1
    for i, option in enumerate(options):
        option_str = f'{i+1:>{number_width}}. {option.label}'
        option.label = option_str
    return options


class Menu(UIComponent):
    def __init__(
            self,
            title: str,
            options: Sequence[MenuItem],
            selected_option: int = 0,
            option_padding: int = 0,
    ) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.title = title
        self.options = options
        self.selected_option = -1
        self.option_padding = option_padding
        self._bbox = (0,0,0,0)

        self.select_option(selected_option)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self._bbox

    def select_option(self, selected_option: int) -> None:
        if 0 <= self.selected_option < len(self.options):
            self.options[self.selected_option].deselect()
        self.selected_option = selected_option
        if 0 <= self.selected_option < len(self.options):
            self.options[self.selected_option].select()

    def select_next(self) -> int:
        if self.selected_option < len(self.options)-1:
            self.select_option(self.selected_option + 1)
        return self.selected_option

    def select_prev(self) -> int:
        if self.selected_option > 0:
            self.select_option(self.selected_option - 1)
        return self.selected_option

    def activate_item(self) -> None:
        if not (0 <= self.selected_option < len(self.options)):
            return
        self.logger.debug(f'activating menu item {self.selected_option} {self.options[self.selected_option]}')
        self.options[self.selected_option].action()

    def select_and_activate_option(self, selected_option:int) -> Any:
        self.select_option(selected_option)
        self.activate_item()

    def draw(self, canvas: interface.BasicCanvas, y: int, x: int) -> None:
        min_y = y
        min_x = x

        max_x = x+len(self.title)+2
        canvas.addstr(
            y, x,
            f' {self.title} \n\n',
            curses.A_UNDERLINE | curses.A_BOLD
        )

        y += 2
        for i, option in enumerate(self.options):
            option.draw(canvas, y, x)
            max_x = max(max_x, option.width)
            y += option.height + self.option_padding

        self._bbox = (min_y, min_x, y, max_x)

    def key_list(self) -> Collection[interface.KeyBinding]:
        nav_help = config.key_help(self, "j")
        act_help = config.key_help(self, "\r")
        select_and_act_help = config.key_help(self, "1")
        key_options = [
            interface.KeyBinding(ord("j"), self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("k"), self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("s"), self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("w"), self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(curses.KEY_DOWN, self.select_next, nav_help, help_key="menu_nav"),
            interface.KeyBinding(curses.KEY_UP, self.select_prev, nav_help, help_key="menu_nav"),
            interface.KeyBinding(ord("\r"), self.activate_item, act_help, help_key="menu_act"),
        ]
        key_options.extend(
            interface.KeyBinding(
                ord(str(x+1)),
                (lambda x=x: self.select_and_activate_option(x)), # type: ignore
                select_and_act_help,
                help_key="menu_select_and_act")
            for x in range(min(len(self.options), 9))
        )
        return key_options

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


class MeterMenu(UIComponent):
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
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.draw_y, self.draw_x, self.draw_y+self.height, self.draw_x+self.width)

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
        self.draw_y = y
        self.draw_x = x
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

def dtmf_sample(number_str: Union[str, dtmf.model.String], sample_rate:int, mark_duration:float=0.03, space_duration:float=0.03, level:float=-6.0, pause_duration:float=0.03) -> npt.NDArray[np.float64]:
    gp = dtmf._generator.GenerationParams(mark_duration=mark_duration, space_duration=space_duration, level=level, pause_duration=pause_duration)
    if isinstance(number_str, str):
        dtmf_string = dtmf.parse(number_str)
    elif isinstance(number_str, dtmf.model.String):
        dtmf_string = number_str
    else:
        raise ValueError(f'number_str only supports str and dtmf.String, not {number_str.__class__}')

    return np.array(list(dtmf.generate(dtmf_string, params=gp, sample_rate=sample_rate)))
