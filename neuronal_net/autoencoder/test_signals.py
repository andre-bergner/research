import numpy as np


def rk4(f, x0, t0, dt, N):
   t = t0 + dt * np.arange(N+1)
   x = np.zeros((N+1, np.size(x0)))
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

def lorenz(n_samples=10000):
   def ode(x,t):
      return np.array([
         10 * (x[1] - x[0]),
         x[0] * (28 - x[2]) - x[1],
         x[0] * x[1] - (8./3.) * x[2]
      ])
   return rk4(ode, np.array([1,0,0]), 0., 0.01, 1000+n_samples-1)[1000:,0] / 20

#def roessler(n_samples=10000):

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
   return xs, ys

def make_training_set_ar(signal_gen, frame_size=30, n_pairs=1000):
   sig = signal_gen(n_pairs + frame_size + 1)
   ns = range(n_pairs)
   xs = np.array([sig[n:n+frame_size] for n in ns])
   ys = np.array([sig[n+frame_size:n+frame_size+1] for n in ns])
   return xs, ys


# [[out1_1, out1_2] , [out2_1, out2_2]] -->  [out1_1+out2_1, out1_2+out2_2]
# [['a1', 'a2'] , ['b1', 'b2']] -->  [out1_1+out2_1, out1_2+out2_2]

def concat(inputs, outputs):
  con_ins = np.concatenate(inputs)
  con_outs = [np.concatenate(x) for x in zip(*outputs)]
  return con_ins, con_outs
