import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import matplotlib.patches as patches
import matplotlib.lines as mlines

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

plt.rcParams['mathtext.fontset'] = 'stix' # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

class Arrow3D(FancyArrowPatch):
   def __init__(self, xs, ys, zs, *args, **kwargs):
      FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
      self._verts3d = xs, ys, zs

   def draw(self, renderer):
      xs3d, ys3d, zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
      self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
      FancyArrowPatch.draw(self, renderer)


fig = plt.figure(figsize=(6,6))


# ---------------------------------------------------------------------
# LATENT SPACE
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------------------

signal = lambda n: 0.6*np.sin(0.05*np.arange(n)) + 0.3*np.sin(np.pi*0.05*np.arange(n))
data = signal(300)

def plot_signal(ax):
   n = np.arange(len(data))
   plt.plot(n, data, '-k', linewidth=0.7)
   plt.plot(n[::6], data[::6], '.k', markersize=4)
   ax.set_yticks([])
   ax.set_ylim(-1.1, 1.1)

ax = fig.add_axes([0.05, 0.57, 0.4, 0.27])
plot_signal(ax)
#ax.text(50, -1.5, r'input window $\mathbf{\xi}_n$', fontsize=16)
ax.bar( 150, 2, 100, -1, linestyle='--', ec='#666666', fc='#66666644' )
ax.xaxis.tick_top()
ax.set_xticklabels([])
ax.set_ylabel(r'$x_n$', fontsize=18)
ax.text(170, 1.25, r'sample window $\mathbf{\xi}$', fontsize=16)

# ---------------------------------------------------------------------
# BACKGROUND
# ---------------------------------------------------------------------

canvas = fig.add_axes([0, -0.4, 1, 1.4])
canvas.patch.set_alpha(0)
canvas.axis('off')
canvas.set_xticks([])
canvas.set_yticks([])


# ---------------------------------------------------------------------
# PERCEPTRONS
# ---------------------------------------------------------------------

def plot_layers(ax, pos, scale, color, dir=1):
   width = 0.1*scale[0]
   height = 0.1*scale[1]
   ax.add_patch(patches.Rectangle( [pos[0]-3*width/2, pos[1] ], 3*width, dir*height, ec='k', fc=color ))
   ax.add_patch(patches.Rectangle( [pos[0]-2*width/2, pos[1] + dir*1.2*height ], 2*width, dir*height, ec='k', fc=color ))
   ax.add_patch(patches.Rectangle( [pos[0]-1*width/2, pos[1] + dir*2.4*height ], 1*width, dir*height, ec='k', fc=color ))

def add_lines(ax, points, color='k', linestyle='-', arrowstyle=None):
   for p1,p2 in zip(points[:-1], points[1:]):
      ax.add_line(mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=linestyle))
   if arrowstyle:
      p1 = np.array(points[-2])
      p2 = np.array(points[-1])
      dp = p2-p1
      canvas.add_patch(patches.FancyArrowPatch(
         p2, p2+0.05*dp, arrowstyle=arrowstyle, mutation_scale=15, ec='k', fc='k'))


# encoder
add_lines(canvas, [[0.25, 0.699], [0.25, 0.68]], arrowstyle='-|>')
plot_layers(canvas, [0.25, 0.68], [0.4, 0.25], '#8899cc', dir=-1)
canvas.text(0.4, 0.63, r'encoder $e(\mathbf{\xi})$', fontsize=16)

# latent space
canvas.add_patch(patches.Rectangle( [0.15, 0.54], 0.2, 0.05, ec='k', fc='#dddddd' ))
add_lines(canvas, [[0.25, 0.59], [0.25, 0.54]], color='k', linestyle='--')
canvas.text(0.18, 0.555, r'$Z_1$', fontsize=20)
canvas.text(0.29, 0.555, r'$Z_2$', fontsize=20)
canvas.text(0.4, 0.555, r'$Z = Z_1 \otimes Z_2$', fontsize=20)

# decoder
plot_layers(canvas, [0.18, 0.45], [0.4, 0.25], '#88cc99', dir=1)
plot_layers(canvas, [0.32, 0.45], [0.4, 0.25], '#88cc99', dir=1)
canvas.text(0.4, 0.485, r'decoders $d_k(\mathbf{z}_k)$', fontsize=16)
canvas.text(0.4, 0.45, r'with $k \in \{1,2\}$', fontsize=14)

canvas.add_patch(patches.Rectangle( [0.22, 0.4], 0.06, 0.04, ec='k', fc='#88cc99' ))
canvas.text(0.234, 0.41, r'$+$', fontsize=20)
add_lines(canvas, [[0.18, 0.449], [0.18, 0.42], [0.22, 0.42]], arrowstyle='-|>')
add_lines(canvas, [[0.32, 0.449], [0.32, 0.42], [0.281, 0.42]], arrowstyle='-|>')


add_lines(canvas, [[0.25, 0.399], [0.25, 0.35], [0.7, 0.35], [0.7, 0.7]], arrowstyle='-|>')
#canvas.text(0.55, 0.74, r'$\mathrm{argmin}\ L(\mathbf{\xi}, \mathbf{\hat\xi})$', fontsize=18)
canvas.text(0.5, 0.745, r'$\mathrm{min}\ L\left(\mathbf{\xi}, \left(\left(d_1+d_2\right) \circ e\right)(\mathbf{\xi})\right)$', fontsize=18)
add_lines(canvas, [[0.25, 0.88], [0.25, 0.95], [0.7, 0.95], [0.7, 0.8]], arrowstyle='-|>')

