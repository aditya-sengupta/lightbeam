# %%
if __name__ == "__main__":
    from lightbeam.prop import tri_solve_vec as tri_py
    import lightbeamrs
    from itertools import repeat
    from copy import copy
    import numpy as np
    from timeit import timeit

    a, b, c, r, g, u = list(np.load(f"../data/trimats_{f}.npy") for f in ['a', 'b', 'c', 'r', 'g', 'u'])
    N = a.shape[0]
    print(timeit(lambda: lightbeamrs.tri_solve_vec(N, a, b, c, r, g, u), number=1))
    u_rust = copy(u)

    print(timeit(lambda: tri_py(a, b, c, r, g, u), number=1))
# %%
import lightbeamrs
import numpy as np
N = 10
a, b, c, r, g, u = [np.random.random((N,N)) + 1j * np.random.random((N,N)) for _ in range(6)]
lightbeamrs.tri_solve_vec(N, a, b, c, r, g, u)
# %%
import numpy as np
from lightbeamrs import zip_testing
Q = (1 + 1j) * np.ones((2,2))
R = (1 + 2j) * np.ones((2,2))
zip_testing(Q, R)
assert np.allclose(Q, R)
# %%
