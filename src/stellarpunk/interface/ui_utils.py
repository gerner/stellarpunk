""" Utils for drawing Stellarpunk UI """

import curses

from stellarpunk import core, interface

def draw_sprite(sprite:core.Sprite, canvas:interface.Canvas, y:int, x:int) -> None:
    y_off = 0
    for row in sprite.text:
        for x_off, s in enumerate(row):
            if (x_off,y_off) in sprite.attr:
                a,c = sprite.attr[x_off,y_off]
                canvas.window.addstr(y+y_off, x+x_off, s, a | curses.color_pair(c))
            else:
                canvas.window.addstr(y+y_off, x+x_off, s)
        y_off+=1

def draw_portrait(portrait:core.Sprite, canvas:interface.Canvas) -> None:
    padding = " " * ((canvas.width-portrait.width)//2)
    for row in portrait.text:
        canvas.window.addstr(padding+f'{row}\n')

def draw_heading(canvas:interface.Canvas, text:str) -> None:
    canvas.window.addstr("\t")
    canvas.window.addstr(f' {text} ', curses.A_UNDERLINE|curses.A_BOLD)
    canvas.window.addstr("\n\n")
