""" Displays an image in the terminal. """

import sys

import imageio.v3 as iio # type: ignore
import numpy as np
import drawille # type: ignore

#im = iio.imread('/home/nick/downloads/stellarpunk/graphics/79portraits/79 1bit-portraits.png')
im = iio.imread(sys.argv[1])

#portrait_pos = (6,7)
#limits = (
#        (portrait_pos[0]*32, portrait_pos[0]*32+32)
#        (portrait_pos[1]*32, portrait_pos[1]*32+32)
#)
limits = ((0,im.shape[1]),(0,im.shape[0]))

c = drawille.Canvas()
for x in range(*limits[0]):
    for y in range(*limits[1]):
        # ndimage is column then row
        #if im[y,x,3] < 128:
        if im[y,x] > 128:
            c.set(x,y)

top_padding = 3
left_padding = 8

height = 24
width = 48

blank_line = " "*width
left_space = " "*left_padding

for _ in range(top_padding):
    print(blank_line)

for row in c.rows():
    print(f'{left_space}{row:<{width-left_padding}}')

for _ in range(height - len(c.rows()) - top_padding):
    print(blank_line)
