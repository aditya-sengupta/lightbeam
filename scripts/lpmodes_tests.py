
import matplotlib.pyplot as plt
rcore = 21.8/2
ncore = 1.4504
nclad = 1.4504 - 5.5e-3

import hcipy as hc

V = get_V(2*np.pi,rcore,ncore,nclad)
modes = get_modes(V)

for mode in modes:

    xa = ya = np.linspace(-15,15,1000)
    xg , yg = np.meshgrid(xa,ya)

    if mode[0]==0:
        print(mode)
        lp= lpfield(xg,yg,mode[0],mode[1],rcore,1,ncore,nclad,'cos')
        lp /= np.max(lp)

        f = hc.Field(lp.flatten() , hc.make_pupil_grid((1000,1000),diameter=30) )

        hc.imshow_field(f)
        plt.show()
        continue
    else:

        print(mode,"cos")
        lp= lpfield(xg,yg,mode[0],mode[1],rcore,1,ncore,nclad,'cos')
        lp /= np.max(lp)

        f = hc.Field(lp.flatten() , hc.make_pupil_grid((1000,1000),diameter=30) )

        hc.imshow_field(f)
        plt.show()

        print(mode,'sin')
        lp= lpfield(xg,yg,mode[0],mode[1],rcore,1,ncore,nclad,'sin')
        lp /= np.max(lp)

        f = hc.Field(lp.flatten() , hc.make_pupil_grid((1000,1000),diameter=30) )

        hc.imshow_field(f)
        plt.show()
        
'''
rcore = 4   
wl = np.linspace(1.2,1.4,100)
k0 = 2*np.pi/wl
ncore = 1.4504
nclad = 1.4504 - 5.5e-3

modenumes = np.vectorize(get_num_modes)(2*np.pi/wl,rcore,ncore,nclad)
import matplotlib.pyplot as plt
plt.plot(wl,modenumes)
plt.show()

#k = 2*np.pi/0.98
#print(get_NA(1.4504 + 0.0088,1.4504))
'''

"""
rcore = 2.2
NA = 0.16
wl0 = 1.55
ncore = 4
nclad = np.sqrt(ncore*ncore-NA*NA)


print(nclad,ncore)

k0 = 2*np.pi/wl0

print(get_MFD(k0,rcore,ncore,nclad))
"""
"""
import matplotlib.pyplot as plt

wl = 1.
k = 2*np.pi/wl
ncore = 1.4504
nclad = 1.4504 - 5.5e-3

rcore = 12
V = get_V(k,rcore,ncore,nclad)

modes = get_modes(V)
print(modes)
xa = ya = np.linspace(-20,20,801)
xg, yg = np.meshgrid(xa,ya)


fig,axs = plt.subplots(7,6)

for mode in modes:
if mode[0] == 0:
    field = lpfield(xg,yg,mode[0],mode[1],rcore,wl,ncore,nclad)
    axs[0,2*mode[1]-2].imshow(np.real(field),vmin = -np.max(np.real(field)),vmax = np.max(np.real(field)))
else:
    fieldcos = lpfield(xg,yg,mode[0],mode[1],rcore,wl,ncore,nclad,'cos')
    fieldsin = lpfield(xg,yg,mode[0],mode[1],rcore,wl,ncore,nclad,'sin')
    axs[mode[0],2*mode[1]-2].imshow(np.real(fieldcos),vmin = -np.max(np.real(fieldcos)),vmax = np.max(np.real(fieldcos)))
    axs[mode[0],2*mode[1]-1].imshow(np.real(fieldsin),vmin = -np.max(np.real(fieldsin)),vmax = np.max(np.real(fieldsin)))

for _axs in axs:
for ax in _axs:
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

plt.subplots_adjust(hspace=0,wspace=0)

plt.show()
"""