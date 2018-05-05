import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import matplotlib.patches as patches

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
   def __init__(self, xs, ys, zs, *args, **kwargs):
      FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
      self._verts3d = xs, ys, zs

   def draw(self, renderer):
      xs3d, ys3d, zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
      self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
      FancyArrowPatch.draw(self, renderer)


fig = plt.figure(figsize=(8,4))



# ---------------------------------------------------------------------
# TORUS IN THE LATENT SPACE
# ---------------------------------------------------------------------

N = 100

theta = np.linspace(0, 2.*np.pi, N)
phi = np.linspace(0, 2.*np.pi, N)
theta, phi = np.meshgrid(theta, phi)
c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

ls = LightSource(azdeg=0, altdeg=65)
# Shade data, creating an rgb array.
rgb = ls.shade(z, plt.cm.gray)

ax = fig.add_axes([0.63, 0.2, 0.35, 0.6], projection='3d')
ax.set_zlim(-3,3)


K = 430
tx, ty, tz = np.zeros(K), np.zeros(K), np.zeros(K)
for k in range(K):
    theta = -0.6 + 0.3*k/2
    phi = 1.3 + 0.02*k/2
    tx[k] = (c + a*np.cos(theta)) * np.cos(phi)
    ty[k] = (c + a*np.cos(theta)) * np.sin(phi)
    tz[k] = a * np.sin(theta)
ax.plot(tx[:-1], ty[:-1], tz[:-1], 'k', alpha=0.8, linewidth=2)
a = Arrow3D(tx[-2:], ty[-2:], tz[-2:], mutation_scale=10, lw=.5, alpha=0.7, arrowstyle="-|>", color="k")
ax.add_artist(a)

ax.plot_surface(x, y, z, rstride=1, cstride=1, color='#aaaabbee', linewidth=0)
ax.view_init(36, -26)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel(r'$z_1$', fontsize=14)
ax.set_ylabel(r'$z_2$', fontsize=14)
ax.set_zlabel(r'$z_3$', fontsize=14)
ax.xaxis.labelpad = -12
ax.yaxis.labelpad = -12
ax.zaxis.labelpad = -12




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


ax = fig.add_axes([0.05, 0.1, 0.34, 0.35])
plot_signal(ax)
ax.text(50, -1.5, r'input window $\mathbf{\xi}_n$', fontsize=16)
ax.bar( 125, 2, 100, -1, linestyle='--', ec='#445599', fc='#8899cc66' )
ax.xaxis.tick_top()
ax.set_xticklabels([])
ax.set_ylabel(r'$x_n$', fontsize=16)

ax = fig.add_axes([0.05, 0.55, 0.34, 0.35])
plot_signal(ax)
ax.set_xticks([0,100,200,300])
ax.set_xlabel(r'$n$', fontsize=16)
ax.xaxis.set_label_coords(0.5, -0.07)
ax.set_ylabel(r'$x_n$', fontsize=16)

ax.text(100, 1.3, r'target window $\mathbf{\xi}_{n\!+\!1}$', fontsize=16)
ax.bar( 150, 2, 100, -1, linestyle='--', ec='#449955', fc='#88cc9966' )
#ax.plot([100, 100], [-1,1], '--', color='#449955')
#ax.plot([100, 200], [1,1], '--', color='#449955')
#ax.plot([200, 200], [-1,1], '--', color='#449955')
#ax.plot([100, 200], [-1,-1], '--', color='#449955')



# ---------------------------------------------------------------------
# PERCEPTRON
# ---------------------------------------------------------------------

ax = fig.add_axes([0.23, 0.1, 0.45, 0.8])
ax.patch.set_alpha(0)
ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])

# encoder
ax.add_patch(patches.Arrow( 0.05, 0.25, 0.43, 0.0, width=0.1, ec='k', fc='#8899ccaa' ))
#ax.add_patch(patches.FancyArrowPatch( [0., 0.3], [0.48, 0.3], arrowstyle='->', ec='k', fc='#8899cc' ))
ax.add_patch(patches.Rectangle( [0.5, 0.05], 0.04, 0.4, ec='k', fc='#8899cc' ))
ax.add_patch(patches.Rectangle( [0.56, 0.125], 0.04, 0.25, ec='k', fc='#8899cc' ))
ax.add_patch(patches.Rectangle( [0.62, 0.2], 0.04, 0.1, ec='k', fc='#8899cc' ))
ax.text(0.56, 0.05, r'encoder $e(\mathbf{\xi}_n)$', fontsize=16)

# decoder
ax.add_patch(patches.Arrow( 0.48, 0.75, -0.36, 0.0, width=0.1, ec='k', fc='#88cc99aa' ))
ax.add_patch(patches.Rectangle( [0.5, 0.55], 0.04, 0.4, ec='k', fc='#88cc99' ))
ax.add_patch(patches.Rectangle( [0.56, 0.625], 0.04, 0.25, ec='k', fc='#88cc99' ))
ax.add_patch(patches.Rectangle( [0.62, 0.7], 0.04, 0.1, ec='k', fc='#88cc99' ))
ax.text(0.56, 0.9, r'decoder $d(\mathbf{z}_n)$', fontsize=16)

# z-arc
ax.add_patch(patches.FancyArrowPatch(
   [0.7, 0.25], [0.7, 0.75], connectionstyle='arc3, rad=1.1',
   mutation_scale=20, ec='k', fc='#999999'
))

ax.text(1.1, 0.0, 'latent space', fontsize=16)