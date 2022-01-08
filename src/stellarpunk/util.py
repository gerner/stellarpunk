""" Utility methods broadly applicable across the codebase. """

import math

import numpy as np

def fullname(o):
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

def human_distance(distance_meters):
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
        sector_loc_x, sector_loc_y,
        meters_per_char_x, meters_per_char_y):
    """ converts from sector coord to drawille coord. """

    # characters are 2x4 (w,h) "pixels" in drawille
    return (
            int((sector_loc_x) / meters_per_char_x * 2),
            int((sector_loc_y) / meters_per_char_y * 4)
    )

def sector_to_screen(
        sector_loc_x, sector_loc_y,
        ul_x, ul_y,
        meters_per_char_x, meters_per_char_y):
    """ converts from sector coord to screen coord. """
    return (
            int((sector_loc_x - ul_x) / meters_per_char_x),
            int((sector_loc_y - ul_y) / meters_per_char_y)
    )

def screen_to_sector(
        screen_loc_x, screen_loc_y,
        ul_x, ul_y,
        meters_per_char_x, meters_per_char_y,
        screen_offset_x=0, screen_offset_y=0):
    """ converts from screen coordinates to sector coordinates. """
    return (
            (screen_loc_x-screen_offset_x) * meters_per_char_x + ul_x,
            (screen_loc_y-screen_offset_y) * meters_per_char_y + ul_y
    )

def magnitude(x, y):
    return math.sqrt(x*x + y*y)

def cartesian_to_polar(x, y):
    r = math.sqrt(x*x + y*y)
    if x == 0:
        if y > 0:
            a = math.pi/2
        else:
            a = -1 * math.pi/2
    else:
        a = math.atan(y/x)
    if x < 0:
        return r, a+math.pi
    elif y < 0:
        return r, a+math.pi*2
    else:
        return r, a

def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return (x,y)


def normalize_angle(angle, shortest=False):
    angle = angle % (2*math.pi)
    angle = (angle + 2*math.pi) if angle < 0 else angle
    if not shortest or angle <= math.pi:
        return angle
    else:
        return angle - 2*math.pi

class NiceScale:
    """ Produces a "nice" scale for a range that looks good to a human.

    from https://stackoverflow.com/a/16959142/553580
    """

    def __init__(self, minv, maxv, maxTicks=10, constrain_to_range=False, tickSpacing=None):
        self.maxTicks = maxTicks
        self.tickSpacing = tickSpacing
        self.lst = 10
        self.niceMin = 0
        self.niceMax = 0
        self.minPoint = minv
        self.maxPoint = maxv
        self.constrain_to_range = constrain_to_range
        self.calculate()

    def calculate(self):
        if self.tickSpacing is None:
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


    def niceNum(self, lst, rround):
        self.lst = lst
        exponent = 0 # exponent of range */
        fraction = 0 # fractional part of range */
        niceFraction = 0 # nice, rounded fraction */

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

    def setMinMaxPoints(self, minPoint, maxPoint):
          self.minPoint = minPoint
          self.maxPoint = maxPoint
          self.calculate()

    def setMaxTicks(self, maxTicks):
        self.maxTicks = maxTicks;
        self.calculate()
