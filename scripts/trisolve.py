# %%
from lightbeam.lightbeamrs import tri_solve_vec
import numpy as np

N = 4096
a, b, c, r, g, u = [np.random.uniform(size=(N,N)) + 1j * np.random.uniform(size=(N,N))  for _ in range(6)]
%timeit tri_solve_vec(a, b, c, r, g, u)
# %%
