""" Displays an image in the terminal. """

import imageio.v3 as iio # type: ignore
import numpy as np
import drawille # type: ignore

im = iio.imread('/home/nick/downloads/stellarpunk/graphics/79portraits/79 1bit-portraits.png')

portrait_pos = (6,7)

c = drawille.Canvas()
for x in range(portrait_pos[0]*32, portrait_pos[0]*32+32):
    for y in range(portrait_pos[1]*32, portrait_pos[1]*32+32):
        # ndimage is column then row
        if im[y,x,3] < 128:
            c.set(x,y)

for row in c.rows():
    print(row)
