''' Example tests showcasing the potential savings in computation time offered by AMR.'''

import matplotlib.pyplot as plt
import numpy as np
from lightbeam import RectMesh3D, make_lant3big, make_lant6_saval, Prop3D, normalize, lpfield, norm_nonu, overlap_nonu

def compute_port_power(ds, AMR=False, ref_val=2e-4, max_iters=5, remesh_every=50):
    # wavelength
    wl = 1.0 #um

    # mesh 
    xw = 64 #um
    yw = 64 #um
    zw = 1000 #um
    num_PML = int(4/ds) # number of cells
    dz = 1

    mesh = RectMesh3D(xw,yw,zw,ds,dz,num_PML)
    mesh.xy.max_iters = max_iters

    xg,yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML] , mesh.yg[num_PML:-num_PML,num_PML:-num_PML]

    taper_factor = 4
    rcore = 2.2/taper_factor # INITIAL core radius
    rclad = 4
    nclad = 1.4504
    ncore = nclad + 0.0088
    njack = nclad - 5.5e-3

    # lant = make_lant3big(rclad/2, rcore, rclad, 0, zw, (ncore, nclad, njack), final_scale=taper_factor)
    lant = make_lant6_saval(rclad/2, rcore, rclad, 0, zw, (ncore, nclad, njack), final_scale=taper_factor)

    def launch_field(x, y):
        return normalize(lpfield(x, y, 0, 1, rclad, wl, ncore, nclad))
        
    # propagation

    prop = Prop3D(wl, mesh, lant, nclad)

    if AMR:
        u = prop.prop2end(launch_field, ref_val=ref_val, remesh_every=remesh_every)
    else:
        u = prop.prop2end(launch_field(xg, yg))

    xg, yg = np.meshgrid(mesh.xy.xa, mesh.xy.ya, indexing='ij')

    w = mesh.xy.get_weights()

    # get the output port powers

    output_powers = []
    for pos in lant.final_core_locs:
        _m = norm_nonu(lpfield(xg-pos[0], yg-pos[1], 0, 1, rcore*taper_factor, wl, ncore, nclad), w)
        output_powers.append(np.power(overlap_nonu(_m, u, w), 2))

    return np.array(output_powers)

if __name__ == "__main__":
    output = compute_port_power(1/16)
    print(output)
