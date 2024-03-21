# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
u = np.load("../u_test.npy")
# %%
fig, axes = plt.subplots()
plt.imshow(np.abs(u), cmap="binary")
c = plt.Circle((145, 145), 120, fill=False)
axes.add_artist(c)
plt.gca().set_axis_off()
plt.savefig("../figures/example_lantern_output.pdf", bbox_inches='tight', pad_inches=0.0)
# %%
