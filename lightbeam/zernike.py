import numpy as np
from scipy.special import factorial,jn,gamma
import matplotlib.pyplot as plt
from numpy import fft
from numpy.polynomial import polynomial as poly

""" functions for working with Zernike modes """

def nm2j(n,m):
    raise Exception("use hcipy.mode_basis.zernike_to_noll instead")

def j2nm(j):
    raise Exception("use hcipy.mode_basis.noll_to_zernike instead")

def Z_rad(n,m):
    """radial component of Zernike mode"""
    m = abs(m)
    indices = np.arange(0,1+int((n-m)/2))
    coeffs = np.power(-1,indices) * factorial(n-indices) / factorial(indices) / factorial( int((n+m)/2) - indices ) / factorial( int((n-m)/2) - indices)
    powers = n - 2 * indices

    return lambda r: np.sum( coeffs[:,None,None] * np.power( np.repeat([r],len(powers),axis=0) , powers[:,None,None] ) , axis = 0 )

def Z_az(n,m):
    """azimuthal component of Zernike mode"""
    def _inner_(theta):
        if m == 0:
            return np.sqrt(n+1)
        elif m>0:
            return np.sqrt(2*(n+1)) * np.cos( m*theta )
        else:
            return np.sqrt(2*(n+1)) * np.sin( abs(m)*theta )
    return _inner_

def Z(n,m):
    """Zernike mode (n,m indexed). returns function of r,theta"""
    assert (n-abs(m))%2 == 0 , "in Z(n,m), n - |m| must be even"
    def _inner_(r,theta):
        return ( Z_rad(n,m)(r) * Z_az(n,m)(theta) ) * (r<=1)
    return _inner_ 

def Zj_pol(j):
    """Zernike mode (j indexed). returns function of r,theta"""
    n,m = j2nm(j)    
    return Z(n,m)

def Zj_cart(j):
    """Zernike mode (j indexed). returns function of x,y"""
    def _inner_(x,y):
        r = np.sqrt(x*x+y*y)
        t = np.arctan2(y,x)
        return Zj_pol(j)(r,t)
    return _inner_

def Qj(j):
    """Fourier transform of Zernike mode"""
    n,m = j2nm(j)
    if j == 1: 
        lim = 1
    else: 
        lim = 0

    def _inner_(k,phi):
        with np.errstate(divide="ignore",invalid="ignore"):
            kdep = np.where( k!=0 , jn(n+1, 2*np.pi*k) / (np.pi*k),lim)
        if m>0:
            return np.sqrt(2*(n+1)) * kdep * np.power(-1,int((n-m)/2)) * np.power(-1j,m) * np.cos(m*phi)
        elif m<0:
            return np.sqrt(2*(n+1)) * kdep * np.power(-1,int((n-m)/2)) * np.power(-1j,m) * np.sin(m*phi)
        else:
            return np.sqrt(n+1) * kdep * np.power(-1,int(n/2))
    return _inner_

def Qj_cart(j):
    def _inner_(kx,ky):
        k = np.sqrt(kx*kx+ky*ky)
        phi = np.arctan2(ky,kx)
        return Qj(j)(k,phi)
    return _inner_

def high_pass(n):
    """ returns a high pass filter that can be applied to any turbulence PSD 
        to remove Zernike modes up to and including order n
    """
    def _inner_(kx,ky):
        filt = np.zeros_like(kx)
        for j in range(1,n+1):
            filt += np.power(np.abs(Qj_cart(j)(kx,ky)),2)
        return 1 - filt
    return _inner_

def low_pass(n):
    """ returns a low pass filter that can be applied to any turbulence PSD
        to only include Zernike modes up to and including order n """
    def _inner_(kx,ky):
        filt = np.zeros_like(kx)
        for j in range(1,n+1):
            filt += np.power(np.abs(Qj_cart(j)(kx,ky)),2)
        return filt
    return _inner_

def noll_cov(i,j,D,r0):
    """covariance matrix for Kolmogorov turbulence"""

    n,m = j2nm(i)
    _n,_m = j2nm(j)

    if m!= _m or (i-j)%2==1:
        return 0 
    
    Kzz_fac = gamma(14/3)* np.power(24/5 * gamma(6/5),5/6) * np.power(gamma(11/6),2) / (2*np.pi**2)
    Kzz = Kzz_fac * np.power(-1, int((n+_n-2*abs(m))/2) ) * np.sqrt((n+1)*(_n+1))
    fac1 = 0.5 * (n + _n - 5/3)
    fac2 = 0.5 * (n - _n  + 17/3)
    fac3 = 0.5 * (_n - n + 17/3)
    fac4 = 0.5 * (n + _n + 23/3)

    return ( Kzz * gamma(fac1) * np.power(D/r0,5/3) ) / (gamma(fac2)*gamma(fac3)*gamma(fac4))

def compute_noll_mat(N,save=True):
    """ compute normalized noll matrix in (D/r0)^5/3 units
        args

        N: number of Zernike modes to use
        save: flag to save matrix to "nollmat.npy"

        returns: Zernike covariance matrix
    """
    mat = np.empty((N,N))
    for i in range(2,N+2):
        for j in range(2,N+2):
            mat[i-2,j-2] = noll_cov(i,j,1,1)
    if save:
        np.save("nollmat",mat)
    return mat

def phase_screen_func(tmat):
    N = tmat.shape[0]
    a = np.random.normal(size=N)
    a = np.dot(tmat,a)

    def _inner_(x,y):
        out = np.zeros_like(x)
        for i in range(N):
            j = i+2
            out += a[i]*Zj_cart(j)(x,y)
        return out
    
    return _inner_

def inner_product(u0,u1,ds):
    """inner product which makes use of orthogonality relation of Zernikes as defined in this file"""
    return np.sum(u0*u1) * ds * ds / np.pi
