# %%
import numpy as np
from matplotlib import pyplot as plt
import lightbeam as lb

wl = 1.55
scale = 8
rcore = 2.2
rclad = 37.6/2*scale
nclad = 1.444 # cladding index
ncore = 1.444 + 0.01036 # lantern core refractive index
njack = np.sqrt(nclad**2 - 0.125**2) # jacket index
core_offset = rclad/2.5 # offset of cores from origin
z_ex = 60000
lant = lb.optics.make_lant19(core_offset,rcore,rclad,0,z_ex, (ncore,nclad,njack),final_scale=1/scale)

mesh = lb.RectMesh3D(
        xw = 512, # um
        yw = 512, # um
        zw = z_ex, # um
        ds = 1, # um
        dz = 2, # um
        PML = 12 # grid units
    )
lant.set_sampling(mesh.xy)

xg, yg = mesh.grids_without_pml()

launch_fields = [
    lb.normalize(lb.lpfield(xg-pos[0], yg-pos[1], 0, 1, rcore, wl, ncore, nclad))
    for pos in lant.init_core_locs]

prop = lb.Prop3D(wl, mesh, lant, nclad)
u = prop.prop2end(launch_fields[5]) # pick an arbitrary off-center one
# u is distorted/not being guided by the cores
# %%
