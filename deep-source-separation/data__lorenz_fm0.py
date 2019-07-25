import test_data as td
import numpy as np

n_samples = 10000

#src1, src2 = 0.3 * td.lorenz, 0.2 * td.fm_strong
src1, src2 = 0.2 * td.lorenz, 0.4 * td.fm_strong0
mix = src1 + src2

np.savetxt('lorenz_fm0.txt', mix(n_samples))
np.savetxt('src1__lorenz.txt', src1(n_samples))
np.savetxt('src2__freqmod.txt', src2(n_samples))
