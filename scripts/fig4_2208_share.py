import numpy as np
from optics import lant6_saval, lant3big
from mesh import RectMesh3D
from prop import Prop3D
import LPmodes
from misc import normalize, norm_nonu, overlap_nonu
import sys
import subprocess
from matplotlib import pyplot as plt

from hcipy import *
from hcipy.mode_basis import zernike_ansi

wl = 1.0 # um
xw = 40 # um
yw = 40 # um
zw = 1000 # um
ds = 1/2
num_PML = 32 # number of cells
dz = 1
taper_factor = 4
rcore = 1.5
rclad = 10
ncore = 1.4504 + 0.0088
njack = 1.4504 - 5.5e-3 # jacket index
nclad = 1.4504

lant = lant6_saval(rcore,rcore,rcore,rcore,rclad,ncore,nclad,njack,rclad*2/3,zw,final_scale=taper_factor)
mesh = RectMesh3D(xw,yw,zw,ds,dz,num_PML)
lant.set_sampling(mesh.xy)
prop = Prop3D(wl,mesh,lant,nclad)

pupil_grid = make_pupil_grid(len(mesh.xg), 1)
focal_grid = make_focal_grid(8, 9.1, reference_wavelength=wl, f_number=17) 
# shaneAO f number
# 9.1 is a magic number idk
fprop = FraunhoferPropagator(pupil_grid, focal_grid)

def save_for_zampl(zern, ampl):
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=wl)
    u_in = np.array(normalize(fprop(wf).electric_field)).reshape(tuple(wf.grid.shape))
    u = prop.prop2end_uniform(u_in)
    # np.save(f"data/2208_4_{zern}_{ampl}.npy", u)
    return u

u_out = save_for_zampl(2, -0.3) # vary 2-6 and -1.0 to 1.0
output_powers = []
t = 2 * np.pi / 5
w = mesh.xy.get_weights()
core_locs = [[0.0,0.0]] + [[2 * np.cos(i*t), 2 * np.sin(i*t)] for i in range(5)]
for pos in core_locs:
    _m = norm_nonu(LPmodes.lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*taper_factor,wl,ncore,nclad),w)
    output_powers.append(np.power(overlap_nonu(_m,u_out,w),2))

print(output_powers)