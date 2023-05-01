# %%
from fig4_2208 import *
from misc import norm_nonu, overlap_nonu
from tqdm import tqdm
import re
# %%
t = 2 * np.pi / 5
core_locs = [[0.0,0.0]] + [[2 * np.cos(i*t), 2 * np.sin(i*t)] for i in range(5)]
xg,yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML] , mesh.yg[num_PML:-num_PML,num_PML:-num_PML]
w = mesh.xy.get_weights()
# %%
u = np.load("data/2208_4_2_-0.7.npy")
output_powers = []
for pos in core_locs:
    _m = norm_nonu(LPmodes.lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*taper_factor,wl,ncore,nclad),w)
    output_powers.append(np.power(overlap_nonu(_m,u,w),2))
# %%
powers = np.empty((5,21,6)) # input zernikes x amplitudes x ports
ampls = np.empty((5,21)) # input zernikes x amplitudes
input_zerns = [2, 3, 4, 5, 6]
files = os.listdir("data")
for z in input_zerns:
    zfiles = list(filter(lambda x: x.startswith(f"2208_4_{z}"), files))
    assert len(zfiles) == 21

for z in input_zerns:
    zfiles = list(filter(lambda x: x.startswith(f"2208_4_{z}"), files))
    for (j, f) in tqdm(enumerate(zfiles)):
        ampl = float(re.match(fr"2208_4_{z}_(\S+).npy", f).group(1))
        ampls[z-2][j] = ampl
        u = np.load(f"data/{f}")
        for (i, pos) in enumerate(core_locs):
            _m = norm_nonu(LPmodes.lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*taper_factor,wl,ncore,nclad),w)
            powers[z-2][j][i] = np.power(overlap_nonu(_m,u,w),2)
# %%
np.save("data/2208_4_ampls.npy", ampls)
np.save("data/2208_4_powers", powers)
# %%
ampls = np.load("data/2208_4_ampls.npy")
powers = np.load("data/2208_4_powers.npy")
powers = np.nan_to_num(powers, 0)
colors = ['r', 'g', 'b', 'k', 'c', 'm']
fig, axs = plt.subplots(5,1, figsize=(2,10), sharex=True)
plt.subplots_adjust(hspace=0.4)
for j in range(5):
    perm = np.argsort(ampls[j])
    for i in range(6):
        axs[j].plot(ampls[j][perm], powers[j,:,i][perm], colors[i], label=f"Port {i}")
    axs[j].set_title(f"Zernike mode {j+2}")
        
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel("Input amplitude")
plt.ylabel("Output response")
plt.savefig("figures/2208_4_repr.pdf", bbox_inches='tight')
# %%
