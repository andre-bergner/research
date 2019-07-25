import numpy as np
from pylab import *
from sklearn.decomposition import FastICA, PCA

def ica(x, n_components, max_iter=1000):
   ica_trafo = FastICA(n_components=n_components, max_iter=max_iter)
   return ica_trafo.fit_transform(x)

t = np.arange(20020)
s1 = np.sin(0.05*t)
s2 = np.sin(0.1121*t)

x = s1 + s2
# embedding = np.stack([ x[0:-4], x[1:-3], x[2:-2], x[3:-1] ])
embedding = np.stack([
   x[0:-16],
   x[1:-15],
   x[2:-14],
   x[3:-13],
   x[4:-12],
   x[5:-11],
   x[6:-10],
   x[7:-9],
   x[8:-8],
   x[9:-7],
   x[10:-6],
   x[11:-5],
   x[12:-4],
   x[13:-3],
   x[14:-2],
   x[15:-1],
])


W = ica(embedding, 16)
cs = embedding.T.dot(W)
