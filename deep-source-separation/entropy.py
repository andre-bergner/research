import numpy as np
import scipy.special
import entropy_estimators

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


def sample_entropy(U, m, r):

   def maxdist(win_n, win_m):
      #result = max([abs(ua - va) for ua, va in zip(win_n, win_m)])
      #return result
      return norm(win_n - win_m)

   def phi(m):
      wins = np.array([[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)])
      C = [
         len([
            1 for j in range(len(wins))
            if i != j and maxdist(wins[i], wins[j]) <= r
         ])
         for i in range(len(wins))
      ]
      return sum(C)

   N = len(U)
    
   return -np.log(phi(m+1) / phi(m))


def naive_mutual_information(x, y, bins=32):

   pxy, *_ = np.histogram2d(x, y, bins=bins)
   pxy /= sum(pxy)
   px = np.sum(pxy, axis=0)
   py = np.sum(pxy, axis=1)
   return np.sum(pxy * np.log(0.000001 + pxy / (0.000001 + np.outer(py,px))))


from scipy.stats import chi2_contingency

def mutual_information(x, y, bins=32):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi


def mutual_info(x, y, k=5):
    return entropy_estimators.mi(
        np.expand_dims(x, axis=-1).tolist(),
        np.expand_dims(y, axis=-1).tolist(), k)
