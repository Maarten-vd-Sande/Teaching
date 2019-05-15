import matplotlib.pyplot as plt
import numpy as np
from tSNE import get_psij
from numba import jit, prange
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons, make_blobs


@jit(nopython=True)
def modularity_np(Aij, c):
    two_m = np.sum(Aij)
    ki = np.sum(Aij, axis=0)
    kikj = ki * np.expand_dims(ki, 1)
    kronecker = (c == np.expand_dims(c, 1))
    np.fill_diagonal(kronecker, 0)

    result = (1 / two_m) * np.sum((Aij - kikj / two_m) * kronecker)
    return result

@jit(nopython=True)
def single_point_iterate(Aij, c, i):
    """ take random i for prettier visual """
    i %= Aij.shape[0]
    results, classes = np.zeros(np.unique(c).shape[0]), np.unique(c)
    for j, cls in enumerate(classes):
        new_c = np.copy(c)
        new_c[i] = cls
        # classes[j] = new_c[i]
        results[j] = modularity_np(Aij, new_c)
    c[i] = classes[np.argmax(results)]
    return c


data, real_clusters = make_moons(n_samples=200, noise=0.05, random_state=0)
data, y_true = make_blobs(n_samples=200, centers=4,
                          cluster_std=2.5, random_state=42)
Aij = get_psij(np.linalg.norm(data[np.newaxis] - data[:, np.newaxis], axis=2)**2, 80)

fig = plt.figure()
ax_scat = fig.add_subplot(1, 1, 1)
ax_scat = ax_scat.scatter(data[:, 0], data[:, 1], s=100)


def update(i):
    global Aij, ax_scat
    groups = single_point_iterate(Aij, c, i)
    print(f"he, {i}")
    ax_scat.set_color([f'C{g%10}' for g in groups])


c = np.arange(Aij.shape[0])
ani = FuncAnimation(fig, update, interval=1, frames=900)
plt.show()
