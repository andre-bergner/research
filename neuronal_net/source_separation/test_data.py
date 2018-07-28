import numpy as np

import sys
sys.path.append('../')
from keras_tools import test_signals as TS

class LazyGenerator():

   def __init__(self, generator):
      self.generator = generator

   def __call__(self, n):
      return self.generator(n)

   @staticmethod
   def _unlift_gen(x, n):
      if isinstance(x, LazyGenerator):
         return x.generator(n)
      else:
         return x

   def __add__(self, rhs): return LazyGenerator(lambda n: self.generator(n) + self._unlift_gen(rhs, n))
   def __radd__(self, lhs): return LazyGenerator(lambda n: self._unlift_gen(lhs, n) + self.generator(n))
   def __sub__(self, rhs): return LazyGenerator(lambda n: self.generator(n) - self._unlift_gen(rhs, n))
   def __rsub__(self, lhs): return LazyGenerator(lambda n: self._unlift_gen(lhs, n) - self.generator(n))
   def __mul__(self, rhs): return LazyGenerator(lambda n: self.generator(n) * self._unlift_gen(rhs, n))
   def __rmul__(self, lhs): return LazyGenerator(lambda n: self._unlift_gen(lhs, n) * self.generator(n))
   def __div__(self, rhs): return LazyGenerator(lambda n: self.generator(n) / self._unlift_gen(rhs, n))
   def __rdiv__(self, lhs): return LazyGenerator(lambda n: self._unlift_gen(lhs, n) / self.generator(n))



sin0 = LazyGenerator(lambda n: np.sin(0.03*np.arange(n)))
sin1 = LazyGenerator(lambda n: np.sin(0.05*np.arange(n)))
sin2 = LazyGenerator(lambda n: np.sin(np.pi*0.05*np.arange(n)))
sin3 = LazyGenerator(lambda n: np.sin(0.021231*np.arange(n)))
exp1 = LazyGenerator(lambda n: np.exp(-0.001*np.arange(n)))
sin1exp = sin1 * exp1
sin2am = sin2 * (1 + 0.4*sin3)
sin2am = LazyGenerator(lambda n: sin2(n) * (1+0.4*np.sin(0.021231*np.arange(n))))
kick1 = LazyGenerator(lambda n: np.sin( 100*np.exp(-0.001*np.arange(n)) ) * np.exp(-0.001*np.arange(n)))
kick2 = LazyGenerator(lambda n: np.sin( 250*np.exp(-0.002*np.arange(n)) ) * np.exp(-0.001*np.arange(n)))
kicks2 = LazyGenerator(lambda n: np.sin( 250*np.exp(-0.002*(np.arange(n)%3000) ) ) * np.exp(-0.001*(np.arange(n)%3000)))
tanhsin1 = LazyGenerator(lambda n: np.tanh(4*np.sin(0.05*np.arange(n))))
fm_soft = LazyGenerator(lambda n: np.sin(0.07*np.arange(n) + 4*np.sin(0.00599291*np.arange(n))))
fm_soft1 = LazyGenerator(lambda n: np.sin(np.pi*0.05*np.arange(n) + 3*np.sin(0.00599291*np.arange(n))))
fm_soft3 = LazyGenerator(lambda n: np.sin(np.pi*0.1*np.arange(n) + 6*np.sin(0.00599291*np.arange(n))))
fm_soft3inv = LazyGenerator(lambda n: np.sin(np.pi*0.1*np.arange(n) - 6*np.sin(0.00599291*np.arange(n))))
fm_soft2 = LazyGenerator(lambda n: np.sin(0.15*np.arange(n) + 18*np.sin(0.00599291*np.arange(n))))
fm_med = LazyGenerator(lambda n: np.sin(0.1*np.arange(n) + 1*np.sin(0.11*np.arange(n))))
fm_strong = LazyGenerator(lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n))))
fm_strong2 = LazyGenerator(lambda n: np.sin(0.06*np.arange(n) + 4*np.sin(0.11*np.arange(n))))
fm_hyper = LazyGenerator(lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)) + 2*np.sin(0.009*np.arange(n))))
lorenz = LazyGenerator(lambda n: TS.lorenz(n, [1,0,0])[::1])
lorenz2 = LazyGenerator(lambda n: TS.lorenz(n+25000, [0,-1,0])[25000:])


# some often used test pairs

two_sin = 0.64*sin1, 0.3*sin2
lorenz_fm = 0.3*lorenz, 0.15*fm_strong
kicks_sin1 = kicks2, 0.3*sin2
kicks_sin2 = kicks2, sin2
