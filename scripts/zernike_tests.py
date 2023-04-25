from lightbeam.zernike import *

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