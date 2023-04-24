import numpy as np
from scipy.special import factorial,jn,gamma
import matplotlib.pyplot as plt
from numpy import fft
from numpy.polynomial import polynomial as poly

""" functions for working with Zernike modes """

def nm2j(n,m):
    """ convert n,m indices to j indices following Noll's prescription """
    offset = 0
    if ( m >= 0 and (n%4 in [2,3]) ) or ( m<=0 and (n%4 in [0,1]) ):
        offset = 1
    
    j = int( n*(n+1)/2 ) + abs(m) + offset
    return j

def j2nm(j):
    """ inverse of nm2j """
    
    _nmin = int (0.5 * (-3 + np.sqrt(8*j+1)))
    _nmax = int (0.5 * (-1 + np.sqrt(8*j+1)))

    _m1 = j - int(_nmin * (_nmin+1)/2) - 1
    _m2 = _m1 + 1
    _m3 = j - int(_nmax * (_nmax+1)/2) - 1
    _m4 = _m3 + 1

    guesses = [ [_nmin,_m1], [_nmin,_m2], [_nmax,_m3], [_nmax,_m4] ] 
    for n,m in guesses:
        if m<0 or (n-m)%2 != 0 or m>n:
            continue  
        if j%2 == 0:
            return (n,m)
        else:
            return (n,-m)   
                
    raise Exception("CATASTROPIC FAILURE IN j2nm()")

def Z_rad(n,m):
    """radial component of Zernike mode"""
    m = abs(m)
    indices = np.arange(0,1+int((n-m)/2))
    coeffs = np.power(-1,indices) * factorial(n-indices) / factorial(indices) / factorial( int((n+m)/2) - indices ) / factorial( int((n-m)/2) - indices)
    powers = n - 2 * indices

    def _inner_(r):
        return np.sum( coeffs[:,None,None] * np.power( np.repeat([r],len(powers),axis=0) , powers[:,None,None] ) , axis = 0 )

    return _inner_

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

        
if __name__ == "__main__":
    
    from screen import PhaseScreenGenerator
    plt.style.use('dark_background')

    names = ["piston","x tilt","y tilt", "defocus", "y astig.", "x astig.", "y coma", "x coma", "y trefoil", "x trefoil", "", ""]

    xa=ya = np.linspace(-1,1,1000)
    xg,yg = np.meshgrid(xa,ya)

    z = Zj_cart(2)(xg,yg)
    plt.imshow(z)
    plt.show()

    _z = Zj_cart(3)(xg,yg)

    print( inner_product(z,z,2/1000))
    print( inner_product(z,_z,2/1000))


    """
    fig,axs = plt.subplots(3,4)

    for ax in axs:
        for _ax in ax:
            _ax.axis("off")

    for i in range(3):
        for j in range(4):
            k = j+1+4*i 
            print(k)
            if names[k-1]=="":
                axs[i,j].set_title(r"$j=$"+str(k))
            else:
                axs[i,j].set_title(r"$j=$"+str(k)+ " , " + names[k-1])

            axs[i,j].imshow(Zj_cart(k)(xg,yg),vmin=-3,vmax=3)
    plt.show()

    D = 10. # [m]  telescope diameter
    p = 10/100 # [m/pix] sampling scale

    # set wind parameters
    vy, vx = 4., 1. # [m/s] wind velocity vector
    T = 0.01 # [s]  sampling interval

    # set turbulence parameters
    r0 = 0.1 # [m]
    wl0 = 1 #[um]
    wl = 1 #[um] 

    cutoff = 41

    nollmat = compute_noll_mat(cutoff)

    chol = np.linalg.cholesky(nollmat)

    screenfunc = phase_screen_func(chol)

    xa = np.linspace(-5,5,128)
    ya = np.linspace(-5,5,128)
    xg,yg = np.meshgrid(xa,ya)
    rg = np.sqrt(xg*xg+yg*yg)

    _s = screenfunc(xg/5,yg/5) * (D/r0)**(5/6)

    fig,ax = plt.subplots()

    ax.axis("off")
    ax.imshow(_s,extent=(-5,5,-5,5))
    plt.show()

    psgen = PhaseScreenGenerator(D, p, vy, vx, T, r0, wl0, wl,filter_func=high_pass(cutoff),filter_scale=D/2)
    _s2 = psgen.generate()
    fig,ax = plt.subplots()

    ax.axis("off")
    ax.imshow(_s2,extent=(-5,5,-5,5))
    plt.show()

    _s3 = (_s + _s2)* (rg/5 <=1)
    plt.imshow(_s3,extent=(-5,5,-5,5))
    plt.show() 
    psgen = PhaseScreenGenerator(D, p, vy, vx, T, r0, wl0, wl)
    _s4 = psgen.generate()
    plt.imshow(_s4,extent=(-5,5,-5,5))
    plt.show()
    """


#phase screen low and high pass testing
"""
seed = 345698

mult = 2

cutoff = 3

psgen_lp = PhaseScreenGenerator(mult*D, mult*p, vy, vx,T,r0,wl0,wl,seed=seed,filter_func=low_pass(cutoff),filter_scale=D/2)
psgen_hp = PhaseScreenGenerator(mult*D, mult*p, vy, vx, T, r0, wl0, wl, seed=seed,filter_func=high_pass(cutoff),filter_scale=D/2)
#psgen = PhaseScreenGenerator(D, p, vy, vx, T, r0, wl0, wl, seed=seed)

print("done")

screen_lp = psgen_lp.generate()
fig,ax = plt.subplots()
circle = plt.Circle((0, 0), D/2,edgecolor='white',ls='dashed',facecolor='None')
ax.add_artist(circle)

plt.imshow(screen_lp,extent = (-D*mult/2,D*mult/2,-D*mult/2,D*mult/2))
plt.show()

screen_hp = psgen_hp.generate()
plt.imshow(screen_hp)
plt.show()
"""

"""
j = 8

xa = np.linspace(-8,8,2048)
ya = np.linspace(-8,8,2048)
xg,yg = np.meshgrid(xa,ya)

zernike = Zj_cart(j)(xg,yg)
"""

"""
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.title("zernike mode "+str(j))
plt.imshow(zernike,extent = (-8,8,-8,8))
plt.show()
"""
"""
kxa = np.linspace(-10,10,201) 
kya = np.linspace(-10,10,201) 
kxg,kyg = np.meshgrid(kxa,kya)

#Q = Qj_cart(j)(kxg,kyg)

filt_pt = high_pass(41)

plt.style.use('dark_background')

plt.imshow(filt_pt(kxg,kyg),cmap="Greys")
plt.show()
"""

"""
fftz = fft.fftshift(fft.fft2(zernike)) * (16/2048)**2 / np.pi

plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title("fft of zernike mode "+str(j))
plt.imshow(np.abs(fftz), extent = (-64,64,-64,64),vmin = 0, vmax = 0.5)
plt.show()

plt.title("analytic ft of zernike mode "+str(j))
plt.imshow(np.abs(Q),extent = (-10,10,-10,10),vmin=0, vmax = 0.5)
plt.show()
"""

#plt.imshow(zernike,extent = (-1,1,-1,1))
#plt.show()