""" Utils for drawing Stellarpunk UI """

import curses

from stellarpunk import core, interface

def draw_portrait(portrait:core.Sprite, canvas:interface.Canvas) -> None:
    padding = " " * ((canvas.width-portrait.width)//2)
    for row in portrait.text:
        canvas.window.addstr(padding+f'{row}\n')

def draw_heading(canvas:interface.Canvas, text:str) -> None:
    canvas.window.addstr("\t")
    canvas.window.addstr(f' {text} ', curses.A_UNDERLINE|curses.A_BOLD)
    canvas.window.addstr("\n\n")
