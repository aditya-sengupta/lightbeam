# %%
import numpy as np
from hcipy import *
from hcipy.atmosphere import *
from tqdm import tqdm


fname = "../data/input_fields_from_test_atm.npy"
def make_input_fields():
    pupil_grid = make_pupil_grid(256, 1)
    focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 8, 16)

    r0 = 0.3
    wavelength = 500e-9
    L0 = 10
    velocity = 3
    height = 0
    stencil_length = 2
    oversampling = 128

    mode_basis = make_zernike_basis(500, 1, pupil_grid, 1)

    layers = []
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer2 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer2)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer3 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer3)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer4 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer4)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer5 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer5)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer6 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer6)

    atmosphere = MultiLayerAtmosphere(layers, False)
    atmosphere.Cn_squared = Cn_squared_from_fried_parameter(1/40, wavelength)
    fprop = FraunhoferPropagator(pupil_grid, focal_grid.scaled(wavelength))

    aperture = make_circular_aperture(1)(pupil_grid)
    wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    from lightbeam import lant6_saval, RectMesh3D, Prop3D
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

    wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    aperture = make_circular_aperture(1)(pupil_grid)
    times = np.linspace(0, 2, 2001)
    input_fields = np.empty((len(times), focal_grid.shape[0], focal_grid.shape[1]), dtype=np.complex128)
    for (i, t) in enumerate(tqdm(times)):
        atmosphere.evolve_until(t)
        wf2 = atmosphere.forward(wf)
        wf2.electric_field *= aperture
        u_in = np.array(fprop(wf2).electric_field).reshape(tuple(focal_grid.shape))
        input_fields[i] = u_in
    np.save("../data/input_fields_from_test_atm.npy", input_fields)
    return input_fields

def get_input_fields():
    input_fields = None
    import os
    if not(os.path.isfile(fname)):
        input_fields = make_input_fields()
    else:
        input_fields = np.load(fname)
    return input_fields

if __name__ == "__main__":
    input_fields = get_input_fields()