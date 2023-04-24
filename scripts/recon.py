# %%

from hcipy import *
from hcipy.atmosphere import *

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

# %%
assert False
for t in np.linspace(0, 100, 10):
	atmosphere.evolve_until(t)
	wf2 = atmosphere.forward(wf)
	wf2.electric_field *= aperture
	img = Field(prop(wf2).intensity, focal_grid)

	plt.clf()
	plt.subplot(1,2,1)
	imshow_field(wf2.phase, cmap='RdBu')
	plt.subplot(1,2,2)
	imshow_field(np.log10(img / img.max()), vmin=-6)
	plt.draw()
	plt.pause(0.00001)
# %%
from fig4_2208 import lant, prop
# really really bad design to have "prop" mean both the submodule/file *and* the propagator object!
ntimes = 11
output_powers = np.empty((ntimes, 6))
for (k, t) in enumerate(np.linspace(0, 1, ntimes)):
    atmosphere.evolve_until(0)
    wf2 = atmosphere.forward(wf)
    wf2.electric_field *= aperture
    u = np.array(fprop(wf2).electric_field).reshape(focal_grid.shape)
    ul = prop.prop2end_uniform(normalize(u))
    core_locs = [[0.0,0.0]] + [[2 * np.cos(i*t), 2 * np.sin(i*t)] for i in range(5)]
    for (j, pos) in enumerate(core_locs):
        _m = norm_nonu(LPmodes.lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*taper_factor,wl,ncore,nclad),w)
        output_powers[k][j] = np.power(overlap_nonu(_m,ul,w),2)
# %%
output_powers
# %%
from scipy import linalg as la
# %%
def make_unitary_matrix(N):
    H = np.random.random((N,N)) + 1j * np.random.random((N,N))
    H += H.conjugate().T
    return la.expm(1j*H)
# %%
np.random.seed(100)
U = make_unitary_matrix(6)
K = U @ U.conjugate().T
fig, axs = plt.subplots(2,2)
for axr in axs:
    for ax in axr:
        ax.set_xticks([])
        ax.set_yticks([])
axs[0][0].imshow(U.real)
axs[0][0].set_title("U, real")
axs[0][1].imshow(U.imag)
axs[0][1].set_title("U, imag")
axs[1][0].imshow(K.real)
axs[1][0].set_title("UU*, real")
im = axs[1][1].imshow(K.imag)
axs[1][1].set_title("UU*, imag")
fig.colorbar(im)
# %%
