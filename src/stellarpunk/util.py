""" Utility methods broadly applicable across the codebase. """

from __future__ import annotations

import io
import sys
import math
import bisect
import logging
import pdb
import curses
import re
import collections
from typing import Any, List, Tuple, Optional, Callable, Sequence, Iterable, Mapping, MutableMapping, Union, overload, Deque, Collection

import numpy as np
import numpy.typing as npt
from numba import jit # type: ignore
import drawille # type: ignore

logger = logging.getLogger(__name__)

def fullname(o:Any) -> str:
    # from https://stackoverflow.com/a/2020083/553580
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    if isinstance(o, type):
        klass = o
    else:
        klass = o.__class__

    module = klass.__module__
    if module is None or module == str.__class__.__module__:
        return klass.__qualname__  # Avoid reporting __builtin__
    else:
        return module + '.' + klass.__qualname__

def throttled_log(timestamp:float, throttle:float, logger:logging.Logger, level:int, message:str, limit:float) -> float:
    if timestamp > throttle:
        logger.log(level, message)
        return timestamp + limit
    else:
        return throttle

RE_CAMEL_TO_SNAKE_PHASE_1 = re.compile(r'(.)([A-Z][a-z]+)')
RE_CAMEL_TO_SNAKE_PHASE_2 = re.compile(r'([a-z0-9])([A-Z])')
def camel_to_snake(name: str) -> str:
    name = RE_CAMEL_TO_SNAKE_PHASE_1.sub(r'\1_\2', name)
    return RE_CAMEL_TO_SNAKE_PHASE_2.sub(r'\1_\2', name).lower()

def peaked_bounded_random(
        r:np.random.Generator, mu:float, sigma:float,
        size:Optional[Union[int, Sequence[int]]]=None,
        lb:float=0., ub:float=1.0) -> Union[float, npt.NDArray[np.float64]]:
    if mu <= lb or mu >= ub:
        raise ValueError(f'mu={mu} must be lb<mu<ub')
    if sigma <= 0.:
        raise ValueError(f'sigma={sigma} must be > 0.')

    scale = ub-lb
    mu = (mu-lb)/scale
    sigma = (sigma-lb)/scale
    phi = mu * (1-mu)/(sigma**2.)-1.
    if phi <= 1./mu or phi <= 1./(1.-mu):
        raise ValueError(f'sigma={sigma} must be s.t. after transforming mu and sigma to 0,1, mu * (1-mu)/(sigma**2.)-1. < 1/mu and < 1/(1-mu)')
    alpha = mu * phi
    beta = (1-mu) * phi
    # make sure alpha/beta > 1, which makes beta unimodal between 0,1
    assert alpha > 1.
    assert beta > 1.

    return lb+scale*r.beta(alpha, beta, size=size)

def human_si_scale(value:float, unit:str) -> str:
    if abs(value) < 0:
        return f'{value/1e3:.2f}k{unit}'
    elif abs(value) < 1e3:
        return f'{value:.0f}{unit}'
    elif abs(value) < 1e6:
        return f'{value/1e3:.2f}k{unit}'
    elif abs(value) < 1e9:
        return f'{value/1e6:.2f}M{unit}'
    else:
        return f'{value/1e9:.2f}G{unit}'

def human_distance(distance_meters:float) -> str:
    """ Human readable approx string for distance_meters.

    e.g. 1353 => "1.35km" 353.8 => 353m
    """
    if abs(distance_meters) < 1e3:
        return f'{distance_meters:.0f}m'
    elif abs(distance_meters) < 1e6:
        return f'{distance_meters/1e3:.2f}km'
    elif abs(distance_meters) < 1e9:
        return f'{distance_meters/1e6:.2f}Mm'
    elif abs(distance_meters) < 1e12:
        return f'{distance_meters/1e9:.2f}Gm'
    else:
        return f'{distance_meters:.0e}m'

def human_speed(speed_mps:float) -> str:
    """ Human readable approx string for speed_mps.

    e.g. 322.8389 => "322m/s", 2.80 => "2.80m/s", 13589.89 => "13.5km/s"
    """
    abs_speed = abs(speed_mps)
    if speed_mps == 0.:
        return "0m/s"
    elif abs_speed < 0.01:
        return "0.00m/s"
    elif abs_speed < 1e2:
        return f'{speed_mps:.3}m/s'
    elif abs_speed < 1e4:
        return f'{speed_mps:.0f}m/s'
    elif abs_speed < 1e5:
        return f'{speed_mps/1000.:.3}km/s'
    elif abs_speed < 1e7:
        return f'{speed_mps/1000.:.0f}km/s'
    else:
        return f'{speed_mps:.e}m/s'

def human_timespan(timespan_sec:float) -> str:
    if timespan_sec > 3600:
        return f'{timespan_sec/3600:0.2f}hrs'
    elif timespan_sec > 60:
        return f'{timespan_sec/60:0.2f}min'
    elif timespan_sec > 1:
        return f'{timespan_sec:0.2f}sec'
    elif timespan_sec > 0.001:
        return f'{timespan_sec*1000:0.2f}ms'
    else:
        return f'{timespan_sec*1000*1000:0.2}us'

def sector_to_drawille(
        sector_loc_x:float, sector_loc_y:float,
        meters_per_char_x:float, meters_per_char_y:float) -> Tuple[int, int]:
    """ converts from sector coord to drawille coord. """

    # characters are 2x4 (w,h) "pixels" in drawille
    return (
            int((sector_loc_x) / meters_per_char_x * 2),
            int((sector_loc_y) / meters_per_char_y * 4)
    )

#@jit(cache=True, nopython=True, fastmath=True)
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
            (screen_loc_x-screen_offset_x) * meters_per_char_x + (ul_x+meters_per_char_x),
            (screen_loc_y-screen_offset_y) * meters_per_char_y + (ul_y+meters_per_char_y),
    )

@jit(cache=True, nopython=True, fastmath=True)
def circle_bbox(loc:npt.NDArray[np.float64], r:float) -> Tuple[float, float, float, float]:
    return (loc[0]-r, loc[1]-r, loc[0]+r, loc[1]+r)

@jit(cache=True, nopython=True, fastmath=True)
def magnitude_sq(x:float, y:float) -> float:
    return x*x + y*y

@jit(cache=True, nopython=True, fastmath=True)
def distance_sq(s:npt.NDArray[np.float64], t:npt.NDArray[np.float64]) -> float:
    return magnitude_sq((s - t)[0], (s - t)[1])

@jit(cache=True, nopython=True, fastmath=True)
def magnitude(x:float, y:float) -> float:
    return math.sqrt(x*x + y*y)

@jit(cache=True, nopython=True, fastmath=True)
def distance(s:npt.NDArray[np.float64], t:npt.NDArray[np.float64]) -> float:
    return magnitude((s - t)[0], (s - t)[1])

@jit(cache=True, nopython=True, fastmath=True)
def bearing(s:npt.NDArray[np.float64], t:npt.NDArray[np.float64]) -> float:
    course = t - s
    return normalize_angle(math.atan2(course[1], course[0]), shortest=True)

@jit(cache=True, nopython=True, fastmath=True)
def cartesian_to_polar(x:float, y:float) -> tuple[float, float]:
    r = math.hypot(x, y)
    a = math.atan2(y, x)
    return r, a

@jit(cache=True, nopython=True, fastmath=True)
def polar_to_cartesian(r:float, theta:float) -> tuple[float, float]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x,y)


@jit(cache=True, nopython=True, fastmath=True)
def normalize_angle(angle:float, shortest:bool=False) -> float:
    angle = angle % (2*np.pi)
    angle = (angle + 2*np.pi) if angle < 0 else angle
    if not shortest or angle <= math.pi:
        return angle
    else:
        return angle - 2*np.pi

@jit(cache=True, nopython=True, fastmath=True)
def clip(x:float, min_x:float, max_x:float) -> float:
    return min_x if x < min_x else max_x if x > max_x else x

@jit(cache=True, nopython=True, fastmath=True)
def isclose(a:float, b:float) -> bool:
    return abs(a-b) <= (1e-08 + 1e-05 * abs(b))

@jit(cache=True, nopython=True, fastmath=True)
def isclose_flex(a:float, b:float, rtol:float=1e-05, atol:float=1e-08) -> bool:
    # numba gets confused with default parameters sometimes, so we have this
    # "overload"
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

@jit(cache=True, nopython=True, fastmath=True)
def both_almost_zero(v:npt.NDArray[np.float64]) -> bool:
    return isclose(v[0], 0.) and isclose(v[1], 0.)

def pyisclose(a:float, b:float, rtol:float=1e-05, atol:float=1e-08) -> bool:
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

@jit(cache=True, nopython=True, fastmath=True)
def either_nan_or_inf(v:npt.NDArray[np.float64]) -> bool:
    return math.isnan(v[0]) or math.isnan(v[1]) or math.isinf(v[0]) or math.isinf(v[1])

@jit(cache=True, nopython=True, fastmath=True)
def interpolate(x1:float, y1:float, x2:float, y2:float, x:float) -> float:
    """ interpolates the y given x and two points on a line. """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

@jit(cache=True, nopython=True, fastmath=True)
def translate_rect(rect:Tuple[float, float, float, float], p:Union[Tuple[float, float], npt.NDArray[np.float64]]) -> Tuple[float, float, float, float]:
    return rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]

def point_inside_rect(p:Union[Tuple[float, float], npt.NDArray[np.float64]], rect:Tuple[float, float, float, float]) -> bool:
    return rect[0] < p[0] and p[0] < rect[2] and rect[1] < p[1] and p[1] < rect[3]

def intersects(a:Tuple[float, float, float, float], b:Tuple[float, float, float, float]) -> bool:
    """ returns true iff rect a and rect b overlap. """

    # separating axis theorem: if the rectangles do not intersect then a right
    # side will be left of a left side or a top side will be below a bottom
    # side.

    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

def segment_intersects_rect(segment:Tuple[float, float, float, float], rect:Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    """ returns the subsegment that overlaps rect or None if no overlap. """
    # left and right sides
    l = segments_intersect(segment, (rect[0], rect[1], rect[0], rect[3]))
    r = segments_intersect(segment, (rect[2], rect[1], rect[2], rect[3]))
    # top and bottom sides
    t = segments_intersect(segment, (rect[0], rect[1], rect[2], rect[1]))
    b = segments_intersect(segment, (rect[0], rect[3], rect[2], rect[3]))

    subsegment = tuple(x for x in [l,r,t,b] if x is not None)
    assert len(subsegment) <= 2
    if len(subsegment) == 2:
        return tuple(x for p in subsegment for x in p) # type: ignore
    elif len(subsegment) == 1:
        if rect[0] < segment[0] and segment[0] < rect[2] and rect[1] < segment[1] and segment[1] < rect[3]:
            return (subsegment[0][0], subsegment[0][1], segment[0], segment[1])
        else:
            # the other point better be in the rect
            assert rect[0] < segment[2] and segment[2] < rect[2] and rect[1] < segment[3] and segment[3] < rect[3]
            return (subsegment[0][0], subsegment[0][1], segment[2], segment[3])
    elif rect[0] < segment[0] and segment[0] < rect[2] and rect[1] < segment[1] and segment[1] < rect[3]:
        return segment
    else:
        return None

def segments_intersect(a:Tuple[float, float, float, float], b:Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    """ returns true iff segments a and b intersect. """
    # inspired by https://stackoverflow.com/a/565282/553580
    # if the segments are represented as p+r and q+s
    # compute the intersection point between the two segments: p+t*r == q+u*s

    p = a[0:2]
    r = (a[2] - a[0], a[3]-a[1])
    q = b[0:2]
    s = (b[2] - b[0], b[3]-b[1])

    # r X s
    r_cross_s = r[0]*s[1] - s[0]*r[1]

    # check for non parallel
    if r_cross_s != 0.:
        #t = (q − p) × s / r_cross_s
        #u = (q − p) × r / r_cross_s
        qmp = (q[0]-p[0], q[1] - p[1])
        t = (qmp[0] * s[1] - qmp[1] * s[0]) / r_cross_s
        u = (qmp[0] * r[1] - qmp[1] * r[0]) / r_cross_s
        if t >= 0 and t <= 1 and u >= 0 and u <= 1:
            return (p[0]+t*r[0], p[1]+t*r[1])
        else: return None
    # we count colinear as non intersecting
    return None

@jit(cache=True, nopython=True, fastmath=True)
def enclosing_circle(c1:npt.NDArray[np.float64], r1:float, c2:npt.NDArray[np.float64], r2:float) -> Tuple[npt.NDArray[np.float64], float]:
    """ Finds the smallest circle enclosing two other circles.

    courtesey: https://stackoverflow.com/a/36736270/553580
    """
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    #center-center distance
    dc = math.sqrt(dx**2 + dy**2)
    rmin = min(r1, r2)
    rmax = max(r1, r2)
    if rmin + dc < rmax:
        if r1 < r2:
            x = c2[0]
            y = c2[1]
            R = r2
        else:
            x = c1[0]
            y = c1[1]
            R = r1
    else:
        R = 0.5 * (r1 + r2 + dc)
        x = c1[0] + (R - r1) * dx / dc
        y = c1[1] + (R - r1) * dy / dc

    return (np.array((x,y)), R)

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

def drawille_line(start:Sequence[float], end:Sequence[float], meters_per_char_x:float, meters_per_char_y:float, canvas:Optional[drawille.Canvas]=None, step:Optional[float]=None) -> drawille.Canvas:
    if canvas is None:
        canvas = drawille.Canvas()
    course = np.array(end) - np.array(start)
    distance = np.linalg.norm(course)
    if distance == 0:
        d_x, d_y = sector_to_drawille(start[0], start[1], meters_per_char_x, meters_per_char_y)
        canvas.set(d_x, d_y)
        return canvas
    unit_course = course / distance
    if step is None:
        step = 2.*meters_per_char_x
    d = 0.
    while d <= distance:
        point = unit_course * d + start
        d_x, d_y = sector_to_drawille(point[0], point[1], meters_per_char_x, meters_per_char_y)
        canvas.set(d_x, d_y)
        d += step
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

def lines_to_dict(lines:Sequence[str], bounds:Tuple[int, int, int, int], y:int=0, x:int=0) -> Mapping[Tuple[int, int], str]:
    ret:MutableMapping[Tuple[int, int], str] = {}
    for lineno, line in enumerate(lines):
        if lineno+y < bounds[1] or lineno+y >= bounds[3]:
            continue
        for i,c in enumerate(line):
            if c != " " and c!= chr(drawille.braille_char_offset):
                if x+i < bounds[0] or x+i >= bounds[2]:
                    continue
                ret[(lineno+y, x+i)] = c
    return ret

def draw_canvas_at(canvas:drawille.Canvas, screen:curses.window, y:int, x:int, attr:int=0, bounds:Tuple[int, int, int, int]=(0,0,sys.maxsize,sys.maxsize)) -> None:
    """ Draws canvas to screen with canvas origin appearing at y,x.

    Notice the curses based convention of y preceeding x here. """

    for canvas_y,entries in canvas.chars.items():
        screen_y = canvas_y+y
        if screen_y < bounds[1] or screen_y >= bounds[3]:
            continue
        for canvas_x,v in entries.items():
            screen_x = canvas_x+x
            if screen_x < bounds[0] or screen_x >= bounds[2]:
                continue
            # TODO: is this too expensive just to support some debug use cases?
            if isinstance(v, int):
                screen.addch(screen_y, screen_x, chr(drawille.braille_char_offset+v), attr)
            else:
                screen.addch(screen_y, screen_x, v, attr)
    return

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

def elipsis(string:str, max_length:int) -> str:
    if len(string) <= max_length:
        return string
    else:
        #TODO: is using unicode elipsis the right thing to do here?
        return string[:max_length-1] + "…"

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

def compute_uigrid(
        bbox:Tuple[float, float, float, float],
        meters_per_char_x:float, meters_per_char_y:float,
        bounds:Tuple[int, int, int, int],
        max_ticks:int=10,
    ) ->  Tuple[NiceScale, NiceScale, NiceScale, NiceScale, Mapping[Tuple[int, int], str]]:
    """ Materializes a grid, in text, that fits in a bounding box.

    returns a tuple of major/minor x/y tics and the grid itself in text. """

    # choose ticks

    major_ticks_x = NiceScale(
            bbox[0], bbox[2],
            maxTicks=max_ticks, constrain_to_range=True)
    minor_ticks_y = NiceScale(
            bbox[1], bbox[3],
            maxTicks=max_ticks*4, constrain_to_range=True)
    major_ticks_y = NiceScale(
            bbox[1], bbox[3],
            maxTicks=max_ticks, constrain_to_range=True,
            tickSpacing=major_ticks_x.tickSpacing)
    minor_ticks_x = NiceScale(
            bbox[0], bbox[2],
            maxTicks=max_ticks*4, constrain_to_range=True,
            tickSpacing=minor_ticks_y.tickSpacing)

    c = drawille.Canvas()

    # draw the vertical lines
    i = major_ticks_x.niceMin
    while i < bbox[2]:
        j = minor_ticks_y.niceMin
        while j < bbox[3]:
            d_x, d_y = sector_to_drawille(
                    i, j,
                    meters_per_char_x, meters_per_char_y)
            c.set(d_x, d_y)
            j += minor_ticks_y.tickSpacing
        i += major_ticks_x.tickSpacing

    # draw the horizonal lines
    j = major_ticks_y.niceMin
    while j < bbox[3]:
        i = minor_ticks_x.niceMin
        while i < bbox[2]:
            d_x, d_y = sector_to_drawille(
                    i, j,
                    meters_per_char_x, meters_per_char_y)
            c.set(d_x, d_y)
            i += minor_ticks_x.tickSpacing
        j += major_ticks_y.tickSpacing

    # get upper left corner position so drawille canvas fills the screen
    (d_x, d_y) = sector_to_drawille(
            bbox[0], bbox[1],
            meters_per_char_x, meters_per_char_y)
    # draw the grid to the screen
    text = c.rows(d_x, d_y)

    return (
        major_ticks_x,
        minor_ticks_y,
        major_ticks_y,
        minor_ticks_x,
        lines_to_dict(text, bounds=bounds)
    )

def drawille_circle(radius:float, tick_spacing:float, width:float, height:float, meters_per_char_x:float, meters_per_char_y:float, canvas:Optional[drawille.Canvas]=None) -> drawille.Canvas:
    if canvas is None:
        canvas = drawille.Canvas()

    # shortcut if the entire circle falls outside the bbox
    if radius > width and radius > height:
        return canvas

    # perfect arc length is minor tick spacing, but we want an arc length
    # closest to that which will divide the circle into a whole number of
    # pieces divisible by 4 (so the circle dots match the cross)
    if np.pi / (tick_spacing / radius) < 1.:
        # case where we are asked for fewer than 4 ticks in the whole circle
        return canvas
    else:
        theta_tick = 2 * math.pi / (4 * np.round(2 * math.pi / (tick_spacing / radius) / 4))
    # we'll just iterate over a single quadrant and mirror it
    thetas = np.linspace(0., np.pi/2, int((np.pi/2)/theta_tick), endpoint=False)
    for theta in thetas:
        # skip dots that fall on cross
        if np.isclose(theta % (math.pi/2), 0.):
            continue
        dot_x, dot_y = polar_to_cartesian(radius, theta)

        # skip dots outside bbox
        if dot_x > width/2:
            continue
        if dot_y > height/2:
            continue

        d_x, d_y = sector_to_drawille(
                dot_x, dot_y,
                meters_per_char_x, meters_per_char_y)
        canvas.set(d_x, d_y)
        canvas.set(-d_x, d_y)
        canvas.set(-d_x, -d_y)
        canvas.set(d_x, -d_y)

    return canvas

def compute_uiradar(
        center:Tuple[float, float],
        bbox:Tuple[float, float, float, float],
        meters_per_char_x:float, meters_per_char_y:float,
        bounds:Tuple[int, int, int, int],
        max_ticks:int=10
    ) ->  Tuple[NiceScale, NiceScale, NiceScale, NiceScale, Mapping[Tuple[int, int], str]]:
    """ Materializes a radar in text that fits in bbox. """

    # choose ticks based on x size, y is forced to that, but we still want the
    # min/max
    # we also convert to a bbox relative to the center point (so we can get a
    # a major axis on that center point
    # we also want enough rings to fill the bbox (including partial rings on
    # the diagonals)
    major_ticks_x = NiceScale(
            bbox[0]-center[0], bbox[2]-center[0],
            maxTicks=max_ticks, constrain_to_range=True)
    minor_ticks_y = NiceScale(
            bbox[1]-center[1], bbox[3]-center[1],
            maxTicks=max_ticks*4, constrain_to_range=True)
    major_ticks_y = NiceScale(
            bbox[1]-center[1], bbox[3]-center[1],
            maxTicks=max_ticks, constrain_to_range=True,
            tickSpacing=major_ticks_x.tickSpacing)
    minor_ticks_x = NiceScale(
            bbox[0]-center[0], bbox[2]-center[0],
            maxTicks=max_ticks*4, constrain_to_range=True,
            tickSpacing=minor_ticks_y.tickSpacing)

    c = drawille.Canvas()

    # draw a cross over the center point (from center out, with mirroring)
    # horizonal
    i = 0.
    while i < (bbox[2] - bbox[0])/2:
        d_x, d_y = sector_to_drawille(i, 0,
                meters_per_char_x, meters_per_char_y)
        c.set(d_x, d_y)
        c.set(-d_x, d_y)
        i += minor_ticks_x.tickSpacing

    # vertical
    i = 0.
    while i < (bbox[3] - bbox[1])/2:
        d_x, d_y = sector_to_drawille(0, i,
                meters_per_char_x, meters_per_char_y)
        c.set(d_x, d_y)
        c.set(d_x, -d_y)
        i += minor_ticks_y.tickSpacing

    # draw rings centered on the center point
    # should be enough rings to fill bbox (including partial rings 

    # iterate over rings at major tickSpacing
    max_radius = magnitude(bbox[2] - bbox[0], bbox[3] - bbox[1])/2
    for r in np.linspace(major_ticks_x.tickSpacing, max_radius, int(max_radius/major_ticks_x.tickSpacing), endpoint=False):
        drawille_circle(r, minor_ticks_x.tickSpacing, (bbox[2] - bbox[0]), (bbox[3] - bbox[1]), meters_per_char_x, meters_per_char_y, canvas=c)
    # get upper left corner position so drawille canvas fills the screen
    (d_x, d_y) = sector_to_drawille(
            -(bbox[2] - bbox[0])/2, -(bbox[3]-bbox[1])/2,
            meters_per_char_x, meters_per_char_y)

    # draw the grid to the screen
    text = c.rows(d_x, d_y)

    return (
        major_ticks_x,
        minor_ticks_y,
        major_ticks_y,
        minor_ticks_x,
        lines_to_dict(text, bounds=bounds)
    )

def make_rectangle_canvas(rect:Tuple[float, float, float, float], meters_per_char_x:float, meters_per_char_y:float, step:Optional[float]=None, offset_x:float=0., offset_y:float=0.) -> drawille.Canvas:
    if step is None:
        step = 2

    c = drawille.Canvas()
    # draw top and bottom
    x = rect[0]+offset_x
    y1 = rect[1]+offset_y
    y2 = rect[3]+offset_y
    while x <= rect[2]+offset_x:
        d_x, d_y1 = sector_to_drawille(x, y1, meters_per_char_x, meters_per_char_y)
        _, d_y2 = sector_to_drawille(x, y2, meters_per_char_x, meters_per_char_y)
        c.set(d_x, d_y1)
        c.set(d_x, d_y2)
        x += step
    # draw left and right
    x1 = rect[0]+offset_x
    x2 = rect[2]+offset_x
    y = rect[1]
    while y <= rect[3]+offset_y:
        d_x1, d_y = sector_to_drawille(x1, y, meters_per_char_x, meters_per_char_y)
        d_x2, _ = sector_to_drawille(x2, y, meters_per_char_x, meters_per_char_y)
        c.set(d_x1, d_y)
        c.set(d_x2, d_y)
        y += step
    return c

def make_polygon_canvas(vertices:Sequence[Union[Tuple[float, float]|npt.NDArray[np.float64]|Sequence[float]]], meters_per_char_x:float, meters_per_char_y:float, step:Optional[float]=None, offset_x:float=0., offset_y:float=0., bbox:Optional[Tuple[float, float, float, float]]=None) -> drawille.Canvas:
    if step is None:
        step = 2

    c = drawille.Canvas()

    if len(vertices) == 0:
        return c

    last_point = np.array(vertices[-1])
    for i, next_point in enumerate(vertices):
        point = np.array(next_point)
        d = 0.0
        direction = point - last_point
        distance = magnitude(*direction)
        if distance == 0.0:
            continue
        direction = direction/distance
        while d < distance:
            target_point = last_point + direction * d
            d_x, d_y = sector_to_drawille(target_point[0]+offset_x, target_point[1]+offset_y, meters_per_char_x, meters_per_char_y)
            c.set(d_x, d_y)
            #TODO: should be calculated w.r.t. direction and some trig so we
            # get the desired spacing between dots on the screen
            d += step*meters_per_char_x
        last_point = point

    return c


def make_circle_canvas(r:float, meters_per_char_x:float, meters_per_char_y:float, step:Optional[float]=None, offset_x:float=0., offset_y:float=0., bbox:Optional[Tuple[float, float, float, float]]=None) -> drawille.Canvas:
    c = drawille.Canvas()
    assert r >= 0
    if isclose(r, 0.):
        c.set(0,0)
        return c

    # figure out theta bounds given bbox
    if bbox is not None:
        arcs:List[Tuple[float, float]] = []
        # check if bbox outside circle's bounding box
        if bbox[0] >= r or bbox[1] >= r or bbox[2] <= -r or bbox[3] <= -r:
            return c

        # draw 4 arcs, one for each quadrant
        # corresponding to each corner of the box
        # quadrant 1
        if bbox[2] > 0. and bbox[3] > 0.:
            s = max(np.arccos(bbox[2]/r) if bbox[2] < r else 0.,
                    np.arcsin(bbox[1]/r) if bbox[1] > 0 else 0.)
            t = min(np.arcsin(bbox[3]/r) if bbox[3] < r else np.pi/2,
                    np.arccos(bbox[0]/r) if bbox[0] > 0 else np.pi/2)
            arcs.append((s,t))
        if bbox[3] > 0. and bbox[0] < 0.:
            s = max(np.pi-np.arcsin(bbox[3]/r) if bbox[3] < r else np.pi/2,
                    np.arccos(bbox[2]/r) if bbox[2] < 0 else np.pi/2)
            t = min(np.arccos(bbox[0]/r) if bbox[0] > -r else np.pi,
                    np.pi-np.arcsin(bbox[1]/r) if bbox[1] > 0 else np.pi)
            arcs.append((s,t))
        if bbox[0] < 0. and bbox[1] < 0.:
            s = max(np.pi+np.arccos(-bbox[0]/r) if bbox[0] > -r else np.pi,
                    np.pi+np.arcsin(-bbox[3]/r) if bbox[3] < 0 else np.pi)
            t = min(np.pi+np.arcsin(-bbox[1]/r) if bbox[1] > -r else 1.5*np.pi,
                    np.pi+np.arccos(-bbox[2]/r) if bbox[2] < 0 else 1.5*np.pi)
            arcs.append((s,t))
        if bbox[1] < 0. and bbox[2] > 0.:
            s = max(2*np.pi+np.arcsin(bbox[1]/r) if bbox[1] > -r else 1.5*np.pi,
                    2*np.pi-np.arccos(bbox[0]/r) if bbox[0] > 0 else 1.5*np.pi)
            t = min(2*np.pi-np.arccos(bbox[2]/r) if bbox[2] < r else 2*np.pi,
                    2*np.pi+np.arcsin(bbox[3]/r) if bbox[3] < 0 else 2*np.pi)
            arcs.append((s,t))
    else:
        arcs = [(0.0, 2*np.pi)]

    if step is None:
        step = 2/r*meters_per_char_x
    for (s,t) in arcs:
        theta = s
        while theta < t:
            c_x, c_y = polar_to_cartesian(r, theta)
            d_x, d_y = sector_to_drawille(c_x+offset_x, c_y+offset_y, meters_per_char_x, meters_per_char_y)
            c.set(d_x, d_y)
            theta += step

    return c

def choose_argmax(rnd: np.random.Generator, a:npt.NDArray[Any]) -> int:
    flatnonzero = np.flatnonzero(a == a.max())
    if len(flatnonzero) > 1:
        return rnd.choice(flatnonzero)
    else:
        return flatnonzero[0]

def choose_argmin(rnd: np.random.Generator, a:npt.NDArray[Any]) -> int:
    flatnonzero = np.flatnonzero(a == a.min())
    if len(flatnonzero) > 1:
        return rnd.choice(flatnonzero)
    else:
        return flatnonzero[0]

@overload
def update_ema(value_estimate:float, alpha:float, new_value:float) -> float: ...

@overload
def update_ema(value_estimate:npt.NDArray[np.float64], alpha:float, new_value:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

def update_ema(value_estimate:float|npt.NDArray[np.float64], alpha:float, new_value:float|npt.NDArray[np.float64]) -> float|npt.NDArray[np.float64]:
    return alpha * new_value + (1. - alpha) * value_estimate

@overload
def update_vema(value_estimate:float, volume_estimate:float, alpha:float, value:float, volume:float) -> Tuple[float, float]: ...

@overload
def update_vema(value_estimate:npt.NDArray[np.float64], volume_estimate:npt.NDArray[np.float64], alpha:float, value:npt.NDArray[np.float64], volume:npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

@overload
def update_vema(value_estimate:npt.NDArray[np.float64], volume_estimate:float, alpha:float, value:npt.NDArray[np.float64], volume:float) -> Tuple[npt.NDArray[np.float64], float]: ...

def update_vema(value_estimate:float | npt.NDArray[np.float64], volume_estimate:float | npt.NDArray[np.float64], alpha:float, value:float | npt.NDArray[np.float64], volume:float | npt.NDArray[np.float64]) -> Tuple[float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]]:
    """ Update volume weighted moving average parameters (value, volume). """

    value_estimate = alpha * value_estimate + (1-alpha) * value
    volume_estimate = alpha * volume_estimate + (1-alpha) * volume
    return (value_estimate, volume_estimate)

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
        if self.minPoint != self.maxPoint:
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
