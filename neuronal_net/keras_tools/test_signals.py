import numpy as np


def decaying_sinusoids(size, num_signals=200, num_modes=5):

   def make_mode(damp, freq, phase):
      time = np.arange(size)
      return np.cos(phase + freq*time) * np.exp(damp*time)

   signals = [
      np.sum([
         make_mode( -0.1*d, np.pi*f, np.pi*p )
         for d,f,p in np.random.rand(num_modes,3)],
         axis=0)
      for _ in range(num_signals)
   ]

   return np.array(signals) / np.max(np.abs(signals))



def rk4(f, x0, t0, dt, N):
   t = t0 + dt * np.arange(N+1)
   x = np.zeros([N+1] + list(x0.shape))
   x[0] = x0
   for n in range(N):
      xi1 = x[n]
      f1 = f(xi1, t[n])
      xi2 = x[n] + (dt/2.)*f1
      f2 = f(xi2, t[n+1])
      xi3 = x[n] + (dt/2.)*f2
      f3 = f(xi3, t[n+1])
      xi4 = x[n] + dt*f3
      f4 = f(xi4, t[n+1])
      x[n+1] = x[n] + (dt/6.)*(f1 + 2*f2 + 2*f3 + f4)
   return x

def lorenz_all(n_samples=10000, init=[1,0,0]):
   def ode(x, t):
      return np.array([
         10 * (x[1] - x[0]),
         x[0] * (28 - x[2]) - x[1],
         x[0] * x[1] - (8./3.) * x[2]
      ])
   return rk4(ode, np.array(init), 0., 0.01, 1000+n_samples-1)[1000:] / 20

def lorenz(n_samples=10000, init=[1,0,0]):
   return lorenz_all(n_samples, init)[:,0]

def hindmarsh_rose3(n_samples=10000, init=[0,0,3], i0=3., xr=-8./5.):
   a = 1.0
   b = 3.0
   c = 1.0
   d = 5.0
   r = 0.002
   s = 4.
   #i0 = 3. # -10...10
   def ode(x, t):
      nonlocal a
      nonlocal b
      nonlocal c
      nonlocal d
      nonlocal r
      nonlocal s
      nonlocal xr
      nonlocal i0
      return np.array([
         x[1] + x[0]*x[0]*(b - a*x[0]) - x[2] + i0,
         c - d*x[0]*x[0] - x[1],
         r * (s*(x[0]-xr) - x[2]),
      ])
   return rk4(ode, np.array(init), 0., 0.01, 1000+n_samples-1)[1000:] / 20


def hindmarsh_rose4(n_samples=10000, init=[1,0,0,0]):
   a = 1.0
   b = 3.0
   c = 1.0
   d = 0.99
   e = 1.01
   f = 5.0128
   g = 0.0278
   h = 1.605
   k = 0.9573
   l = 1.619
   mu = 0.0021
   nu = 0.0009
   ft = 1.0
   i0 = 2.9
   o = 3.1
   p = 3.96
   def ode(x, t):
      nonlocal a
      nonlocal b
      nonlocal c
      nonlocal d
      nonlocal e
      nonlocal f
      nonlocal g
      nonlocal h
      nonlocal k
      nonlocal l
      nonlocal mu
      nonlocal nu
      nonlocal ft
      nonlocal i0
      nonlocal o
      nonlocal p
      return np.array([
         ft * ( a*x[1] + b*x[0]*x[0] - c*x[0]*x[0]*x[0] - d*x[2] ) + i0,
         ft * ( e      - x[1]        - f*x[0]*x[0]      - g*x[3] ),
         mu * ( -x[2]    + p * (x[0] + h)  ),
         nu * ( -k*x[3]  + o * (x[1] + l)  )
      ])
   return rk4(ode, np.array(init), 0., 0.01, 1000+n_samples-1)[1000:,0] / 20


#def roessler(n_samples=10000):

def ginzburg_landau(n_samples=1000, n_nodes=100, beta=0.1+0.5j):

   def ode(x,t):
      gamma = 0.1
      omega = 1
      beta_re = 0.0
      beta_im = -1
      r = x[:,0] * x[:,0] + x[:,1] * x[:,1]
      fx = np.array([
         (gamma-r) * x[:,0] - omega     * x[:,1],
         omega     * x[:,0] + (gamma-r) * x[:,1]
      ]).T
      coupling_re = (fx[:-2,0] + fx[2:,0] - 2*fx[1:-1,0])
      coupling_im = (fx[:-2,1] + fx[2:,1] - 2*fx[1:-1,1])
      fx[1:-1,0] += beta.real * coupling_re  -  beta.imag * coupling_im
      fx[1:-1,1] += beta.imag * coupling_re  +  beta.real * coupling_im

      left_coupling_re = (fx[1,0] - fx[0,0])
      left_coupling_im = (fx[1,1] - fx[0,1])
      fx[0,0] += beta.real * left_coupling_re  -  beta.imag * left_coupling_im
      fx[0,1] += beta.imag * left_coupling_re  +  beta.real * left_coupling_im

      right_coupling_re = (fx[-2,0] - fx[-1,0])
      right_coupling_im = (fx[-2,1] - fx[-1,1])
      fx[-1,0] += beta.real * right_coupling_re  -  beta.imag * right_coupling_im
      fx[-1,1] += beta.imag * right_coupling_re  +  beta.real * right_coupling_im
      return fx

   #result = rk4(ode, np.random.rand(n_nodes,2), 0., 0.05, 1000+n_samples-1)
   result = rk4(ode, np.linspace(0,1,2*n_nodes).reshape([n_nodes,2]), 0., 0.05, 1000+n_samples-1)
   return result[1000:,:]




def make_signal(n_samples=10000):
   t = np.arange(n_samples)
   sin = np.sin
   signal = 4. * sin(0.2*t + 6*sin(0.017*t))
   # signal = 4. * sin(0.2*t + 6*sin(0.017*t) + 4*sin(0.043*t))
   # signal = 4. * sin(0.2*t + 6*sin(0.02*t)) # +  5. * sin(t/3.7 + .3)

   #signal = 5. * sin(t/3.7 + .3) \
   #       + 3. * sin(t/1.3 + .1) \
   #       + 4. * sin(0.2*t + 6*sin(0.02*t))
          #+ 4. * sin(0.7*t + 14*sin(0.1*t))
          #+ 2. * sin(t/34.7 + .7)
   return signal / 20

def make_signal2(n_samples=10000):
   t = np.arange(n_samples)
   sin = np.sin
   return 0.5 * sin(0.7*t + 3*sin(0.026*t))





def make_training_set(signal_gen, frame_size=30, n_pairs=1000, shift=1, n_out=1):
   sig = signal_gen(n_pairs + frame_size + n_out*shift)
   ns = range(n_pairs)
   xs = np.array([sig[n:n+frame_size] for n in ns])
   ys = [np.array([sig[n+k*shift:n+frame_size+k*shift] for n in ns]) for k in range(1,n_out+1)]
   return xs, ys, sig[frame_size:frame_size+n_pairs].reshape(-1,1), sig[frame_size+1:frame_size+1+n_pairs].reshape(-1,1)


# [[out1_1, out1_2] , [out2_1, out2_2]] -->  [out1_1+out2_1, out1_2+out2_2]
# [['a1', 'a2'] , ['b1', 'b2']] -->  [out1_1+out2_1, out1_2+out2_2]

def concat(inputs, outputs):
  con_ins = np.concatenate(inputs)
  con_outs = [np.concatenate(x) for x in zip(*outputs)]
  return con_ins, con_outs




# ------------------------------------------------------------------------------
# Plotting Tools
# ------------------------------------------------------------------------------

import pylab as pl

def imtight(x, cmap='gray'):
   imshow(x, cmap=cmap, aspect='auto')
   ax = pl.gca()
   ax.set_frame_on(False)
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   pl.tight_layout(pad=0)



from mpl_toolkits.mplot3d import Axes3D

def plot3d(x, y, z, *args, **kwargs):
   pl.figure()
   pl.axes(projection='3d')
   pl.plot(x, y, z, *args, **kwargs)

