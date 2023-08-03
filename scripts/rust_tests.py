# %%
import numpy as np
from lightbeamrs import _arc as _arcrs

def _arc(x, y0, y1, r):
    return 0.5 * r**2 * (np.arctan2(y1,x) - np.arctan2(y0,x))

N = 1000000
x, y0, y1 = [np.random.randn(N) for _ in range(3)]
r = 4.0
%timeit _arc(x, y0, y1, r)
%timeit _arcrs(x, y0, y1, r)
# %%
