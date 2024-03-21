''' Example tests showcasing the potential savings in computation time offered by AMR.'''

import matplotlib.pyplot as plt
import numpy as np
from lightbeam import RectMesh3D, Prop3D, normalize, lpfield, norm_nonu, overlap_nonu
from lightbeam.optics import Lantern
from tqdm import trange

ds = 1/4
AMR = False
ref_val = 2e-4
max_iters = 5
remesh_every=50

# wavelength
wl = 0.3 #um

# mesh 
xw = 64 #um
yw = 64 #um
zw = 500 #um
num_PML = int(4/ds) # number of cells
dz = 1

mesh = RectMesh3D(xw,yw,zw,ds,dz,num_PML)
mesh.xy.max_iters = max_iters

xg,yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML] , mesh.yg[num_PML:-num_PML,num_PML:-num_PML]

taper_factor = 4
rcore = 2.2/taper_factor # INITIAL core radius
rclad = 32
nclad = 1.45
ncore = 1.46
njack = 1.4

lant = Lantern([[0,0]], 16, 20, 24, zw, (ncore, nclad, njack))

s = lpfield(xg, yg, 0, 1, rclad, wl, ncore, nclad)
for i in trange(400):
    if i % 2 == 0:
        l = np.random.randint(100)
    else:
        l = 0
    m = np.random.randint(max(l, 1))
    a = np.random.random()
    try:
        s += a * lpfield(xg, yg, l, m, rclad, wl, ncore, nclad)
    except Exception:
        continue
        
plt.imshow(np.abs(np.real(s) + 0.1 * np.random.random(size=s.shape)), cmap="binary")
plt.gca().set_axis_off()
plt.savefig("figures/outofcontext.pdf", bbox_inches='tight', pad_inches=0.0)