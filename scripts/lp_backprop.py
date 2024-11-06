
# %%
V = lb.LPmodes.get_V(2*np.pi,rclad,ncore,nclad)
modes = lb.LPmodes.get_modes(V)
lp_basis = np.array([
    lb.normalize(lb.lpfield(mesh.xg, mesh.yg, l, m, rclad / final_scale, wl, ncore, nclad)).ravel()
    for (l, m) in modes
]).T

# %%
projection = np.conj(lp_basis.T) @ outputs.T
plt.imshow(np.abs(projection))
# %%
def plot_from_flat(row):
    N = int(np.sqrt(len(row)))
    plt.imshow(row.reshape((N,N)))
    plt.show()
# %%
