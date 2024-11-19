import numpy as np
import lightbeam as lb
import sys

wl = 1.55
core_offset = 10 # offset of cores from origin
ncore = 1.4504 + 0.0088 # lantern core refractive index
nclad = 1.4504 # cladding index
njack = 1.4504 - 5.5e-3 # jacket index
rclad = 37//2
rcore = 2.2
final_scale = 8 # tapering factor of lantern
z_ex = 60_000

rcores = [rcore for _ in range(19)]
lant = lb.optics.make_lant19(core_offset,rcores,rclad,0,z_ex, (ncore,nclad,njack),final_scale=1/final_scale)

mesh = lb.RectMesh3D(
        xw = 512, # um
        yw = 512, # um
        zw = 60_000, # um
        ds = 1, # um
        dz = 50, # um
        PML = 8 # grid units
    )
lant.set_sampling(mesh.xy)
xg, yg = mesh.grids_without_pml()

launch_fields = [
    lb.normalize(lb.lpfield(xg-pos[0], yg-pos[1], 0, 1, rcore, wl, ncore, nclad))
    for pos in lant.init_core_locs]

lbprop = lb.Prop3D(wl, mesh, lant, nclad)
outputs = []
for (i, lf) in enumerate(launch_fields):
    print(f"Illuminating core {i}")
    u = lbprop.prop2end(lf)
    outputs.append(u)
        
input_footprint = np.zeros(mesh.xy.shape)
lant.set_IORsq(input_footprint, 60000)
input_mask = input_footprint >= nclad**2
xl, yl = np.where(input_mask)
proj_xmin, proj_xmax = np.min(xl), np.max(xl)
proj_ymin, proj_ymax = np.min(yl), np.max(yl)
np.save(f"data/vary_core_sizes_backwards_19_base.npy", outputs)
