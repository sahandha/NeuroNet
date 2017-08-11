import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _png

def datastream(n, chunksize=1):
    """Returns a generator over "n" random xy positions and rgb colors."""
    for _ in range(n//chunksize):
        xy = 10 * np.random.random((chunksize, 2))
        color = np.random.random((chunksize, 3))
        yield xy, color

def save(fig, filename):
    """We have to work around `fig.canvas.print_png`, etc calling `draw`."""
    renderer = fig.canvas.renderer
    with open(filename, 'w') as outfile:
        _png.write_png(renderer._renderer.buffer_rgba(),renderer.width, renderer.height,outfile,dpi=fig.dpi)


# We'll be saving the figure's background, so let's make it transparent.
fig, ax = plt.subplots(facecolor='none')

# You'll have to know the extent of the input beforehand with this method.
ax.axis([0, 10, 0, 10])

# We need to draw the canvas before we start adding points.
fig.canvas.draw()

# This won't actually ever be drawn. We just need an artist to update.
col = ax.scatter([5], [5], color=[0.1, 0.1, 0.1], alpha=0.3)

for xy, color in datastream(int(1e6), chunksize=int(1e4)):
    col.set_offsets(xy)
    col.set_color(color)
    ax.draw_artist(col)

save(fig, 'test.png')
