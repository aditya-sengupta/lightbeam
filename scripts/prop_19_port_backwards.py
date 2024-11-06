# %%
import numpy as np
from matplotlib import pyplot as plt
import lightbeam as lb
import hcipy as hc
from hcipy import imshow_field
from tqdm import trange

wl = 1.0
core_offset = 10 # offset of cores from origin
ncore = 1.4504 + 0.0088 # lantern core refractive index
nclad = 1.4504 # cladding index
njack = 1.4504 - 5.5e-3 # jacket index
rclad = 24
rcore = 2.2
final_scale = 4 # tapering factor of lantern
z_ex = 10000
lant = lb.optics.make_lant19(core_offset,rcore,rclad,0,z_ex, (ncore,nclad,njack),final_scale=1/final_scale)

mesh = lb.RectMesh3D(
        xw = 60, # um
        yw = 60, # um
        zw = 10_000, # um
        ds = 0.2, # um
        dz = 5, # um
        PML = 8 # grid units
    )
lant.set_sampling(mesh.xy)
xg, yg = mesh.grids_without_pml()

launch_fields = [
    lb.normalize(lb.lpfield(xg-pos[0], yg-pos[1], 0, 1, rcore, wl, ncore, nclad))
    for pos in lant.init_core_locs]

lbprop = lb.Prop3D(wl, mesh, lant, nclad)
def propagate_backwards():
    outputs = []
    for (i, lf) in enumerate(launch_fields):
        print(f"Illuminating core {i}")
        u = lbprop.prop2end(lf)
        outputs.append(u)
    
    np.save("../data/backprop_19.npy", np.array(outputs))
    
input_footprint = np.zeros(mesh.xy.shape)
lant.set_IORsq(input_footprint, 10000)
input_mask = input_footprint >= nclad**2
xl, yl = np.where(input_mask)
proj_xmin, proj_xmax = np.min(xl), np.max(xl)
proj_ymin, proj_ymax = np.min(yl), np.max(yl)

def sanitize_output(x):
    x = x * input_mask
    return x.ravel() / np.linalg.norm(x)
outputs = np.load("../data/backprop_19.npy")
outputs = np.array([sanitize_output(x) for x in outputs])

telescope_diameter = 0.5
pupil_grid = hc.make_pupil_grid(60, telescope_diameter)
aperture = hc.evaluate_supersampled(hc.make_circular_aperture(telescope_diameter), pupil_grid, 6)

coupling_fraction = 1.0 # unaberrated PSF radius / input core radius
core_radius = 6.0e-6 # m
cladding_radius = 6.0e-6 # m; matching Lightbeam, but for now just step-index
target_psf_radius = coupling_fraction * core_radius
spatial_resolution = wl * 1e-6 / telescope_diameter # m/m = fraction
focal_grid = hc.make_focal_grid(q=10, num_airy=15.1, spatial_resolution=spatial_resolution)
prop = hc.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=2 * target_psf_radius / spatial_resolution)
ref_focal_wf = prop(hc.Wavefront(aperture, wavelength=wl * 1e-6))

lightbeam_footprint = np.logical_and(
    np.logical_and(
        np.min(focal_grid.x) * 1e6 - mesh.xg <= 1e-8,
        np.max(focal_grid.x) * 1e6 - mesh.xg >= -1e-8
    ),
    np.logical_and(
        np.min(focal_grid.y) * 1e6 - mesh.yg <= 1e-8,
        np.max(focal_grid.y) * 1e6 - mesh.yg >= -1e-8
    )
)
projector = np.linalg.inv(outputs @ outputs.T) @ outputs
# %%
def zernike_to_phase(zernike, amplitude):
    if isinstance(zernike, list) and isinstance(amplitude, list):
        phase = hc.Field(sum(a * hc.mode_basis.zernike_ansi(z, D=telescope_diameter)(pupil_grid).shaped for (z, a) in zip(zernike, amplitude)).ravel(), pupil_grid)
    else:
        phase = amplitude * hc.mode_basis.zernike_ansi(zernike, D=telescope_diameter)(pupil_grid)
    return phase

def phase_to_focal(phase):
    aberration = np.exp(1j * phase)
    wavefront = hc.Wavefront(aperture * aberration, wavelength=wl*1e-6)
    focal_field = prop(wavefront)
    return focal_field    

def zernike_to_focal(zernike, amplitude):
    return phase_to_focal(zernike_to_phase(zernike, amplitude))

def lantern_output(focal_field):
    profile_to_project = np.zeros(mesh.xg.shape, dtype=np.complex128)
    profile_to_project[lightbeam_footprint] = focal_field.electric_field.ravel() / np.linalg.norm(focal_field.electric_field)
    coeffs = projector @ profile_to_project.ravel()
    projected = (coeffs @ outputs).reshape((317,317))
    lantern_reading = sum(c * lf for (c, lf) in zip(coeffs, launch_fields))
    return coeffs, projected, lantern_reading

def show_lantern_output(zernike, amplitude):
    if not isinstance(zernike, list):
        zernike = [zernike]
        amplitude = [amplitude]
    phase = hc.Field(sum(a * hc.mode_basis.zernike_ansi(z, D=telescope_diameter)(pupil_grid).shaped for (z, a) in zip(zernike, amplitude)).ravel(), pupil_grid)
    focal_field = phase_to_focal(phase)
    coeffs, projected, lantern_reading = lantern_output(focal_field)
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f"Lantern response for Zernike {zernike}, amplitude {amplitude}")
    fig.subplots_adjust(top=1.4, bottom=0.0)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(phase.shaped)
    axs[0].set_title("Phase screen")
    axs[1].imshow(np.abs(focal_field.intensity.shaped))
    axs[1].set_title("Lantern input")
    axs[2].imshow(np.abs(lantern_reading))
    axs[2].set_title("Lantern output")
    plt.show()
    
def lantern_reading(zernike, amplitude):
    coeffs, _, _ = lantern_output(zernike_to_focal(zernike, amplitude))
    return np.abs(coeffs) ** 2
# %%
poke_amplitude = 1e-10
pokes = []
for i in trange(1, 20):
    amp_plus = lantern_reading(i, poke_amplitude)
    amp_minus = lantern_reading(i, -poke_amplitude)
    pokes.append((amp_plus - amp_minus) / (2 * poke_amplitude))

# %%
int_matrix = np.array(pokes).T
cmd_matrix = np.linalg.pinv(int_matrix, rcond=1e-5)
flat_amp = lantern_reading(1, 0.0)

# %%
nzern = 6
amplitudes = np.arange(-0.1, 0.1001, 0.02)
linearity_responses = np.zeros((nzern, len(amplitudes), 19))
for z in trange(1, nzern+1):
    for (j,a) in enumerate(amplitudes):
        linearity_responses[z-1,j,:] = cmd_matrix @ (lantern_reading(z,a) - flat_amp)
# %%
fig, axs = plt.subplots(2, 3)
for i in range(nzern):
    r, c = i // 3, i % 3
    for j in range(nzern):
        alpha = 1 if i == j else 0.1
        axs[r][c].plot(amplitudes, linearity_responses[i,:,j], alpha=alpha)
        axs[r][c].set_ylim([min(amplitudes), max(amplitudes)])
# %%
