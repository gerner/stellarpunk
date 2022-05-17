""" Utility methods broadly applicable across the codebase. """

from __future__ import annotations

import sys
import math
import bisect
import logging
import pdb
import curses
from typing import Any, Tuple, Optional, Callable, Sequence, Iterable

import numpy as np
from numba import jit # type: ignore
import drawille # type: ignore

def fullname(o:Any) -> str:
    # from https://stackoverflow.com/a/2020083/553580
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__qualname__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__qualname__

def human_distance(distance_meters:float) -> str:
    """ Human readable approx string for distance_meters.

    e.g. 1353 => "1.35km" 353.8 => 353m
    """
    if abs(distance_meters) < 1e3:
        return f'{distance_meters:.0f}m'
    elif abs(distance_meters) < 1e6:
        return f'{distance_meters/1e3:.0f}km'
    elif abs(distance_meters) < 1e9:
        return f'{distance_meters/1e6:.0f}Mm'
    elif abs(distance_meters) < 1e12:
        return f'{distance_meters/1e9:.0f}Gm'
    else:
        return f'{distance_meters:.0e}m'

def sector_to_drawille(
        sector_loc_x:float, sector_loc_y:float,
        meters_per_char_x:float, meters_per_char_y:float) -> Tuple[int, int]:
    """ converts from sector coord to drawille coord. """

    # characters are 2x4 (w,h) "pixels" in drawille
    return (
            int((sector_loc_x) / meters_per_char_x * 2),
            int((sector_loc_y) / meters_per_char_y * 4)
    )

def sector_to_screen(
        sector_loc_x:float, sector_loc_y:float,
        ul_x:float, ul_y:float,
        meters_per_char_x:float, meters_per_char_y:float) -> Tuple[int, int]:
    """ converts from sector coord to screen coord. """
    return (
            int((sector_loc_x - ul_x) / meters_per_char_x),
            int((sector_loc_y - ul_y) / meters_per_char_y)
    )

def screen_to_sector(
        screen_loc_x:int, screen_loc_y:int,
        ul_x:float, ul_y:float,
        meters_per_char_x:float, meters_per_char_y:float,
        screen_offset_x:int=0, screen_offset_y:int=0) -> Tuple[float, float]:
    """ converts from screen coordinates to sector coordinates. """
    return (
            (screen_loc_x-screen_offset_x) * meters_per_char_x + ul_x,
            (screen_loc_y-screen_offset_y) * meters_per_char_y + ul_y
    )

@jit(cache=True, nopython=True)
def magnitude(x:float, y:float) -> float:
    return np.sqrt(x*x + y*y)

@jit(cache=True, nopython=True)
def cartesian_to_polar(x:float, y:float) -> tuple[float, float]:
    r = np.sqrt(x*x + y*y)
    if x == 0:
        if y > 0:
            a = np.pi/2
        else:
            a = -1 * np.pi/2
    else:
        a = np.arctan(y/x)
    if x < 0:
        return r, a+np.pi
    elif y < 0:
        return r, a+np.pi*2
    else:
        return r, a

@jit(cache=True, nopython=True)
def polar_to_cartesian(r:float, theta:float) -> tuple[float, float]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x,y)


@jit(cache=True, nopython=True)
def normalize_angle(angle:float, shortest:bool=False) -> float:
    angle = angle % (2*np.pi)
    angle = (angle + 2*np.pi) if angle < 0 else angle
    if not shortest or angle <= math.pi:
        return angle
    else:
        return angle - 2*np.pi

@jit(cache=True, nopython=True)
def clip(x:float, min_x:float, max_x:float) -> float:
    return min_x if x < min_x else max_x if x > max_x else x

@jit(cache=True, nopython=True)
def isclose(a:float, b:float, rtol:float=1e-05, atol:float=1e-08) -> bool:
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

#@jit(cache=True, nopython=True)
def interpolate(x1:float, y1:float, x2:float, y2:float, x:float) -> float:
    """ interpolates the y given x and two points on a line. """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def intersects(a:Tuple[float, float, float, float], b:Tuple[float, float, float, float]) -> bool:
    """ returns true iff a and b overlap. """

    # separating axis theorem: if the rectangles do not intersect then a right
    # side will be left of a left side or a top side will be below a bottom
    # side.

    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

def drawille_vector(x:float, y:float, canvas:Optional[drawille.Canvas]=None, tick_size:int=3) -> drawille.Canvas:
    """ Draws a vector (x,y) on a drawille canvas and returns it.

    x and y are expressed as drawille canvas coordinates. """

    # draw the vector as an arrow
    if canvas is None:
        canvas = drawille.Canvas()

    r, theta = cartesian_to_polar(x,y)

    # draw tail of arrow
    r_i = r
    while r_i >= 0:
        x_i, y_i = polar_to_cartesian(r_i, theta)
        canvas.set(x_i, y_i)
        r_i -= tick_size

    # draw head of arrow
    for i in range(2):
        x_i, y_i = polar_to_cartesian(r - tick_size/2*i, theta + 0.05*i)
        canvas.set(x_i, y_i)
        x_i, y_i = polar_to_cartesian(r - tick_size/2*i, theta - 0.05*i)
        canvas.set(x_i, y_i)

    return canvas

def draw_line(y:int, x:int, line:str, screen:curses.window, attr:int=0, bounds:Tuple[int, int, int, int]=(0,0,sys.maxsize,sys.maxsize)) -> None:
    if y < bounds[1] or y >= bounds[3]:
        return
    for i, c in enumerate(line):
        # don't write spaces or the braille empty character
        if c != " " and c != chr(drawille.braille_char_offset):
            if x+i < bounds[0] or x+i >= bounds[2]:
                continue
            screen.addch(y, x+i, c, attr)

def draw_canvas_at(canvas:drawille.Canvas, screen:curses.window, y:int, x:int, attr:int=0, bounds:Tuple[int, int, int, int]=(0,0,sys.maxsize,sys.maxsize)) -> None:
    """ Draws canvas to screen with canvas origin appearing at y,x.

    Notice the curses based convention of y preceeding x here. """

    # find the bounds of the canvas in characters
    minrow = min(canvas.chars.keys())
    maxrow = max(canvas.chars.keys())
    mincol = min(min(x.keys()) for x in canvas.chars.values())
    maxcol = max(max(x.keys()) for x in canvas.chars.values())

    rows = canvas.rows()
    for i, row in enumerate(rows):
        y_row = y+minrow+i
        x_row = x+mincol
        if y_row < 0:
            continue
        if x_row < 0:
            if len(row) < -1*x_row:
                continue
            draw_line(y_row, 0, row[-1*x_row:], screen, attr=attr, bounds=bounds)
        else:
            draw_line(y_row, x_row, row, screen, attr=attr, bounds=bounds)

def tab_complete(partial:str, current:str, options:Iterable[str]) -> str:
    """ Tab completion of partial given sorted options. """

    options = sorted(options)
    if not current:
        current = partial

    i = bisect.bisect(options, current)
    if i == len(options):
        return partial
    if options[i].startswith(partial):
        return options[i]
    return partial

def tab_completer(options:Iterable[str])->Callable[[str, str], str]:
    options = list(options)
    def completer(partial:str, command:str)->str:
        p = partial.split(' ')[-1]
        c = command.split(' ')[-1]
        o = tab_complete(p, c, options) or p
        logging.debug(f'p:{p} c:{c} o:{o}')
        return " ".join(command.split(' ')[:-1]) + " " + o
    return completer

class NiceScale:
    """ Produces a "nice" scale for a range that looks good to a human.

    from https://stackoverflow.com/a/16959142/553580
    """

    def __init__(self, minv:float, maxv:float, maxTicks:int=10, constrain_to_range:bool=False, tickSpacing:float=0.) -> None:
        self.maxTicks = maxTicks
        self.tickSpacing = tickSpacing
        self.lst = 10.
        self.niceMin = 0.
        self.niceMax = 0.
        self.minPoint = minv
        self.maxPoint = maxv
        self.constrain_to_range = constrain_to_range
        self.calculate()

    def calculate(self) -> None:
        if self.tickSpacing == 0.:
            self.lst = self.niceNum(self.maxPoint - self.minPoint, False)
            self.tickSpacing = self.niceNum(self.lst / (self.maxTicks - 1), True)

        if self.constrain_to_range:
            self.niceMin = math.ceil(self.minPoint / self.tickSpacing) * self.tickSpacing
            assert self.niceMin >= self.minPoint
            self.niceMax = math.floor(self.maxPoint / self.tickSpacing) * self.tickSpacing
            assert self.niceMax <= self.maxPoint
        else:
            self.niceMin = math.floor(self.minPoint / self.tickSpacing) * self.tickSpacing
            self.niceMax = math.ceil(self.maxPoint / self.tickSpacing) * self.tickSpacing


    def niceNum(self, lst:float, rround:bool) -> float:
        self.lst = lst
        exponent = 0. # exponent of range */
        fraction = 0. # fractional part of range */
        niceFraction = 0. # nice, rounded fraction */

        exponent = math.floor(math.log10(self.lst));
        fraction = self.lst / math.pow(10, exponent);

        if (rround):
            if (fraction < 1.5):
                niceFraction = 1
            elif (fraction < 3):
                niceFraction = 2
            elif (fraction < 7):
                niceFraction = 5;
            else:
                niceFraction = 10;
        else :
            if (fraction <= 1):
                niceFraction = 1
            elif (fraction <= 2):
                niceFraction = 2
            elif (fraction <= 5):
                niceFraction = 5
            else:
                niceFraction = 10

        return niceFraction * math.pow(10, exponent)

    def setMinMaxPoints(self, minPoint:float, maxPoint:float) -> None:
          self.minPoint = minPoint
          self.maxPoint = maxPoint
          self.calculate()

    def setMaxTicks(self, maxTicks:int) -> None:
        self.maxTicks = maxTicks;
        self.calculate()

class PDBManager:
    def __init__(self) -> None:
        self.logger = logging.getLogger(fullname(self))

    def __enter__(self) -> PDBManager:
        self.logger.info("entering PDBManager")

        return self

    def __exit__(self, e:Any, m:Any, tb:Any) -> None:
        self.logger.info("exiting PDBManager")
        if e is not None:
            self.logger.info(f'handling exception {e} {m}')
            print(m.__repr__(), file=sys.stderr)
            pdb.post_mortem(tb)
