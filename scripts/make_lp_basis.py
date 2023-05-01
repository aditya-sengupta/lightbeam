import numpy as np
from lightbeam.LPmodes import lpfield

if __name__ == "__main__":
    N = 160
    a = 3
    modes = [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    r = np.linspace(-a, a, N)
    xg, yg = np.meshgrid(r, r)
    all_modes = np.zeros((len(modes), N ** 2), dtype=np.complex128)
    for (i, (l, m)) in enumerate(modes):
        all_modes[i,:] = lpfield(xg, yg, l, m, 3, 1, 1.5, 1.45).ravel()

    np.save("./data/lp_basis.npy", all_modes)