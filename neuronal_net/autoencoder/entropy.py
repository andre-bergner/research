import numpy as np
import scipy.special


def drop_column(X, n):
   n_row, n_col = X.shape
   if n == 0:
      return X[:,1:]
   if n == n_col-1:
      return X[:,:-1]
   return np.concatenate([X[:,:n].T, X[:,n+1:].T]).T


def victor_entropy(X):

   # Note [d,n] = size(X) (d dims, n samples).
   #
   # J Victor (2002) "Binless strategies for
   # estimation of information from neural
   # data", Physical
   # Use the machine
   # calculations.

   precision = 1e-16

   d, n = X.shape

   # Calculate the nearest neighbor for each point.
   min_dist = np.zeros(n)

   for i in range(n):
      # distance of i'the vector to all others
      X_diff = drop_column(X, i) - X[:,i]#np.repeat( X[:,i], n-1, axis=1)

      # Calculate the minimum Euclidean distance.
      min_dist[i] = np.min(np.sqrt(np.sum(X_diff**2, axis=0)))

      # Ensure the distance is not zero.
      if min_dist[i] < precision:
         min_dist[i] = precision

   # The "Euler-Mascheroni" constant.
   em = 0.5772156649015

   # Calculate area the of an d-dimensional sphere.
   area = d * np.pi**(d/2) / scipy.special.gamma(1 + d/2)

   # Calculate an estimate of entropy based on the
   # mean nearest neighbor distance using an equation
   # from the above citation.
   K = np.log2(area) + np.log2((n-1) / d) + em/np.log(2)
   H = d * np.mean(np.log2(min_dist)) + K
   return np.mean(np.log2(min_dist)) + np.log2(2*(n-1) / d) + em/np.log(2)

   return H
