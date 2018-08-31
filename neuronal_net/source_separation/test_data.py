import numpy as np

import sys
sys.path.append('../')
from keras_tools import test_signals as TS
from keras_tools import wavfile as W

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



def make_sin_gen(freq):
   return LazyGenerator(lambda n: np.sin(freq * np.arange(n)))

sin0 = make_sin_gen(0.03)
sin1 = make_sin_gen(0.05)
sin2 = make_sin_gen(np.pi*0.05)
sin3 = make_sin_gen(0.021231)
sin4 = make_sin_gen(np.pi*0.3)
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
fm_strong0 = LazyGenerator(lambda n: np.sin(0.01*np.arange(n) + 4*np.sin(0.044*np.arange(n))))
fm_strong = LazyGenerator(lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n))))
fm_strong2 = LazyGenerator(lambda n: np.sin(0.06*np.arange(n) + 4*np.sin(0.11*np.arange(n))))
fm_hyper = LazyGenerator(lambda n: np.sin(0.02*np.arange(n) + 4*np.sin(0.11*np.arange(n)) + 2*np.sin(0.009*np.arange(n))))
lorenz = LazyGenerator(lambda n: TS.lorenz(n, [1,0,0])[::1])
lorenz2 = LazyGenerator(lambda n: TS.lorenz(n+25000, [0,-1,0])[25000:])

cello_wav = W.loadwav("./wav/Kawai-K5000W-Cello-C2.wav")
#cello = LazyGenerator(lambda n: cello_wav[2000:2000+n,0])
#cello = LazyGenerator(lambda n: cello_wav[2000+np.linspace(0,1.31*n, n, dtype=int),0])
cello = LazyGenerator(lambda n: cello_wav[np.linspace(0,2*n, n, dtype=int),0])
#bassoon_wav = W.loadwav("./wav/E-Mu-Proteus-FX-Bassoon-C3.wav")
clarinet_wav = W.loadwav("./wav/Ensoniq-SQ-1-Clarinet-C4.wav")
#clarinet = LazyGenerator(lambda n: clarinet_wav[2000:2000+n,0])
clarinet = LazyGenerator(lambda n: clarinet_wav[np.linspace(0,n//2, n, dtype=int),0])

cello_dis3_wav = W.loadwav("./wav/cello5/4_D#3.wav")
cello_dis3 = LazyGenerator(lambda n: cello_dis3_wav[:n,0])
choir_e4_wav = W.loadwav("./wav/boychoir e4.wav")
choir_e4 = LazyGenerator(lambda n: choir_e4_wav[:n,0])


# some often used test pairs

two_sin = 0.64*sin1, 0.3*sin2
lorenz_fm = 0.3*lorenz, 0.15*fm_strong
kicks_sin1 = kicks2, 0.3*sin2
kicks_sin2 = kicks2, sin2
fm_twins = fm_soft3, fm_soft3inv





