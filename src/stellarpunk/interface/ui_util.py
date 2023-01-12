""" Utils for drawing Stellarpunk UI """

import curses
from typing import List, Callable, Any
from dataclasses import dataclass

from stellarpunk import core, interface


def draw_sprite(
        sprite: core.Sprite, canvas: interface.Canvas, y: int, x: int) -> None:
    y_off = 0
    for row in sprite.text:
        for x_off, s in enumerate(row):
            if (x_off, y_off) in sprite.attr:
                a, c = sprite.attr[x_off, y_off]
                canvas.window.addstr(
                    y+y_off, x+x_off,
                    s,
                    a | curses.color_pair(c)
                )
            else:
                canvas.window.addstr(y+y_off, x+x_off, s)
        y_off += 1


def draw_portrait(portrait: core.Sprite, canvas: interface.Canvas) -> None:
    padding = " " * ((canvas.width-portrait.width)//2)
    for row in portrait.text:
        canvas.window.addstr(padding+f'{row}\n')


def draw_heading(canvas: interface.Canvas, text: str) -> None:
    canvas.window.addstr("\t")
    canvas.window.addstr(f' {text} ', curses.A_UNDERLINE | curses.A_BOLD)
    canvas.window.addstr("\n\n")


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

    def draw_menu(self, canvas: interface.Canvas, y: int, x: int) -> None:
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
