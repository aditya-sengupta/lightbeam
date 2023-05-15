from lightbeam.prop import tri_solve_vec as tri_py
import lightbeamrs
from itertools import repeat
from copy import copy

a, b, c, r, g, u = list(np.load(f"data/trimats_{f}.npy")[0,:] for f in ['a', 'b', 'c', 'r', 'g', 'u'])
N = a.size
lightbeamrs.tri_solve_vec(N, a, b, c, r, g, u)

a, b, c, r, g, u = list(map(lambda m: np.reshape(m, (1, N)), [a, b, c, r, g, u]))
tri_py(a, b, c, r, g, u)