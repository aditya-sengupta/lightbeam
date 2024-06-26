import numpy as np

from itertools import count
from numpy.lib import scimath
from scipy.special import jn_zeros, jn, kn, jv, kv
from scipy.optimize import brentq

from hcipy import ModeBasis, Field

def get_NA(ncore,nclad):
    return np.sqrt(ncore*ncore - nclad*nclad)

def get_V(k0,rcore,ncore,nclad):
    return k0 * rcore * get_NA(ncore,nclad)

def get_MFD(k0,rcore,ncore,nclad):
    """Marcuse approx. for straight step index fiber"""
    V = get_V(k0,rcore,ncore,nclad)
    return 2 * rcore * (0.65 + 1.619/np.power(V,1.5) + 2.879/np.power(V,6))

def get_MFD_from_NA(k0,rcore,ncore,NA):
    nclad = np.sqrt(ncore*ncore - NA*NA)
    return get_MFD(k0,rcore,ncore,nclad)

def get_modes(V):
    '''frequency cutoff occurs when b(V) = 0.  solve eqn 4.19.
    checks out w/ the function in the fiber ipynb'''

    l = 0
    m = 1
    modes = []
    while True:

        if l == 0:
            #solving dispersion relation leads us to the zeros of J_1
            #1st root of J_1 is 0. 

            modes.append((0,1))

            while jn_zeros(1,m)[m-1]< V:

                modes.append((l,m+1))
                m+=1
        else:
            #solving dispersion relation leads us to the zeros of J_l-1, neglecting 0
            if jn_zeros(l-1,1)[0]>V:
                break
            
            while jn_zeros(l-1,m)[m-1]<V:
                modes.append((l,m))
                m+=1
        m = 1
        l += 1
    return modes

def get_mode_cutoffs(l, mmax):
    if l > 0:
        return jn_zeros(l-1, mmax)
    else:
        if mmax > 1:
            return np.concatenate(((0.,),jn_zeros(l-1, mmax-1)))
        else:
            return np.array((0.,))

def findBetween(solve_fn, lowbound, highbound, args=(), maxj=10):
    v = [lowbound, highbound]
    s = [solve_fn(lowbound, *args), solve_fn(highbound, *args)]
    
    if s[0] == 0.: return lowbound
    if s[1] == 0.: return highbound

    for j in count():  # probably not needed...
        if j == maxj:
            print("findBetween: max iter reached")
            return v[np.argmin(np.abs(s))]
            #return np.nan
        for i in range(len(s)-1):
            a, b = v[i], v[i+1]
            fa, fb = s[i], s[i+1]

            if (fa > 0 and fb < 0) or (fa < 0 and fb > 0):
                z = brentq(solve_fn, a, b, args=args)
                fz = solve_fn(z, *args)
                if abs(fa) > abs(fz) < abs(fb):  # Skip discontinuities
                    return z

        ls = len(s)
        for i in range(ls-1):
            a, b = v[2*i], v[2*i+1]
            c = (a + b) / 2
            v.insert(2*i+1, c)
            s.insert(2*i+1, solve_fn(c, *args))

def get_b(l, m, V):
    if l == 0:
        def solve_fn(b, V):
            v = V*np.sqrt(b)
            u = V*np.sqrt(1.-b)
            return (u * jn(1, u) * kn(0, v) - v * jn(0, u) * kn(1, v))
    else:
        def solve_fn(b, V):
            v = V*np.sqrt(b)
            u = V*np.sqrt(1.-b)
            return (u * jn(l - 1, u) * kn(l, v) + v * jn(l, u) * kn(l - 1, v))

    epsilon = 1.e-12
    
    Vc = get_mode_cutoffs(l+1, m)[-1]
    if V < Vc:
        lowbound = 0.
    else:
        lowbound = 1.-(Vc/V)**2    
    Vcl = get_mode_cutoffs(l, m)[-1]
    if V < Vcl: 
        return np.nan

    highbound = 1.-(Vcl/V)**2

    if np.isnan(lowbound): lowbound = 0.
    if np.isnan(highbound): highbound = 1.

    lowbound = np.max((lowbound-epsilon,0.+epsilon))
    highbound = np.min((highbound+epsilon,1.))
    b_opt = findBetween(solve_fn, lowbound, highbound, maxj=10, args=(V,))

    return b_opt

def lpfield(xg, yg, l, m, a, wl0, ncore, nclad):
    '''calculate transverse field distribution of lp mode'''

    # assert which in ("cos","sin"), "lp mode azimuthal component is either a cosine or sine, choose either 'cos' or 'sin'"

    V = get_V(2*np.pi/wl0, a, ncore, nclad)
    rs = np.sqrt(np.power(xg,2) + np.power(yg,2))
    b = get_b(l, m, V)
    if np.isnan(b):
        b = 0.001

    u = V * np.sqrt(1 - b)
    v = V * np.sqrt(b) 

    fieldout = np.zeros_like(rs, dtype = np.complex128)
    
    inmask = np.nonzero(rs <= a)
    outmask = np.nonzero(rs > a)

    fieldout[inmask] = jn(l, u * rs[inmask] / a)
    fieldout[outmask] = jn(l, u) / kn(l,v) * kn(l, v * rs[outmask] / a)

    #cosine/sine modulation
    phis = np.arctan2(yg,xg)
    fieldout *= np.exp(1j * l * phis)

    return fieldout

def make_complex_lp_basis(grid, a, wl0, ncore, nclad):
    """
    Make a basis of complex LP modes for a step-index fiber.
    """
    # replacement for hcipy's make_lp_basis, with the complex component
    V = get_V(2*np.pi / wl0, a, ncore, nclad)
    modes = get_modes(V)
    return ModeBasis([
        Field(
            lpfield(grid.x, grid.y, l, m, a, wl0, ncore, nclad),
            grid
        )
        for (l, m) in modes
    ])

def get_IOR(wl):
    """ for fused silica """
    wl2 = wl*wl
    return np.sqrt(0.6961663 * wl2 / (wl2 - 0.0684043**2) + 0.4079426 * wl2 / (wl2 - 0.1162414**2) + 0.8974794 * wl2 / (wl2 - 9.896161**2) + 1)

def get_num_modes(k0,rcore,ncore,nclad):
    V = get_V(k0,rcore,ncore,nclad)
    modes = get_modes(V)
    num = 0
    for mode in modes:
        if mode[0] == 0:
            num += 1
        else:
            num += 2
    return num

def get_all_bs(l, V, bmax):

    def solve_fn(b,V):
        v = V * scimath.sqrt(b)
        u = V * scimath.sqrt(1.-b)

        if l == 0:
            return np.abs(u * jv(1, u) * kv(0, v) - v * jv(0, u) * kv(1, v))
        else:
            return np.abs(u * jv(l - 1, u) * kv(l, v) + v * jv(l, u) * kv(l - 1, v))
    
    from scipy.optimize import fsolve,minimize

    ret = minimize(solve_fn,[-131],args=(V,),bounds = [(None,bmax)]).x



    #return fsolve(solve_fn,-2,args=(V,))
