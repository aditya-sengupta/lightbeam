# %%
import numpy as np
from optics import lant6_saval, lant3big
from mesh import RectMesh3D
from prop import Prop3D
import LPmodes
from misc import normalize
from zernike import Zj_cart
import sys
import subprocess

from hcipy import *
from hcipy.atmosphere import *

# %%
wl = 1.0 # um
xw = 256 # um
yw = 256 # um
zw = 1000 # um
ds = 1 / 4
num_PML = 32 # number of cells
dz = 1
taper_factor = 4
rcore = 1.5
rclad = 8
ncore = 1.4504 + 0.0088
njack = 1.4504 - 5.5e-3 # jacket index
nclad = 1.4504

pupil_grid = make_pupil_grid(256, 1)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 8, 16)
fprop = FraunhoferPropagator(pupil_grid, focal_grid.scaled(wl))

#%%

lant = lant6_saval(rcore,rcore,rcore,rcore,rclad,ncore,nclad,njack,rclad*2/3,zw,final_scale=taper_factor)
mesh = RectMesh3D(xw,yw,zw,ds,dz,num_PML)
xg,yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML] , mesh.yg[num_PML:-num_PML,num_PML:-num_PML]
lant.set_sampling(mesh.xy)
prop = Prop3D(wl,mesh,lant,nclad)


def save_for_zampl(zern, ampl):
    phase = Zj_cart(zern)(mesh.xg / 32, mesh.yg / 32)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=wl)
    u = prop.prop2end_uniform(normalize(fprop(wf).electric_field))
    np.save(f"data/2208_4_{zern}_{ampl}.npy", u)

# %%
out = np.zeros(mesh.xy.shape)
lant.set_IORsq(out,1000)
# %%
if __name__ == "__main__":
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')
    zern = int(sys.argv[1])
    if len(sys.argv) > 2:
        ampls = [float(sys.argv[2])]
    else:
        ampls = np.linspace(-1, 1, 21)
    
    for ampl in ampls:
        save_for_zampl(zern, ampl)

# %%
