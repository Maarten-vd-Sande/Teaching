import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import proj3d
from numba import jit, prange
from matplotlib.animation import FuncAnimation
from gen_data import generate_mikado, generate_gaussians, classcols


datapoints = 300
data = generate_mikado(datapoints)
# data = generate_gaussians(datapoints)

# sort the data on class, and calculate the distance (norm) between all points
np_data, class_id = data.sort_values('class').values[:, :3], data.sort_values('class').values[:, 3]
norms_sq = np.linalg.norm(np_data[np.newaxis, :, :] - np_data[:, np.newaxis, :], axis=2) ** 2

PERPLEXITY = 10
@jit(nopython=True, parallel=True)
def estimate_sigmas(norms_sq, target_perplexity, tlower=1e-10, tupper=1e10, tol=1e-4, max_iter=1000):
    sigmas = np.zeros(norms_sq.shape[0])
    for row in prange(norms_sq.shape[0]):
        # skip the diagonal
        norm_sq = np.roll(norms_sq[row], -row)[1:]

        lower, upper = tlower, tupper
        for i in range(max_iter):
            # guess a sigma value
            sigma = (lower + upper) / 2

            # calculate the corresponding conditional p_ji val
            p_ji = np.exp(-norm_sq / (sigma * sigma))
            p_ji = p_ji / np.sum(p_ji)

            # calculate which perplexity value belongs to this sigma
            entropy = -np.sum(p_ji * np.log2(p_ji))
            real_perplexity = 2**entropy

            # if perplexity is "good enough"; break
            if np.abs(real_perplexity - target_perplexity) <= tol:
                break

            # else continue search
            elif real_perplexity > target_perplexity:
                upper = sigma
            else:
                lower = sigma

        sigmas[row] = sigma
    return np.expand_dims(sigmas, 0)


@jit(nopython=True)
def get_psij(norms_sq, perplexity):
    sigmas = estimate_sigmas(norms_sq, perplexity)

    numerator = np.exp(-norms_sq / (sigmas ** 2).T)
    np.fill_diagonal(numerator, 0)
    ps_ij = numerator / np.expand_dims(np.sum(numerator, axis=1), 1)
    return ps_ij


@jit(nopython=True)
def norm_sq_ax2(arr):
    out = np.zeros((arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = (arr[i, j, 0]**2 + arr[i, j, 1]**2)**0.5
    return out


@jit(nopython=True)
def get_qsij(projection):
    norms_sq = norm_sq_ax2(np.expand_dims(projection, 0) - np.expand_dims(projection, 1))

    numerator = 1 / (1 + norms_sq)
    np.fill_diagonal(numerator, 0)
    qs_ij = numerator / np.expand_dims(np.sum(numerator, axis=1), 1)
    return qs_ij


@jit(nopython=True)
def gradient(ps_ij, qs_ij, projection):
    dist_diff = np.expand_dims(ps_ij - qs_ij, 2)
    proj_diff = np.expand_dims(projection, 0) - np.expand_dims(projection, 1)
    inv_norm = 1 / np.expand_dims(1 + norm_sq_ax2(proj_diff), 2)

    grad = 4 * np.sum(dist_diff * proj_diff * inv_norm, axis=1)
    return grad


@jit(nopython=True)
def gradient_descent(projection, ps_ij, iterations=10):
    for i in range(iterations):
        qs_ij = get_qsij(projection)
        grad = gradient(ps_ij, qs_ij, projection)
        projection += grad
    return projection, qs_ij


if __name__ == "__main__":
    # setup the figure
    fig = plt.figure()
    fig.suptitle('t-SNE')

    ax_3D = fig.add_subplot(2, 2, 3, projection='3d')
    ax_3D.scatter(np_data[:, 0], np_data[:, 1], np_data[:, 2], c=[classcols[c_id] for c_id in class_id])
    ax_3D.axis('equal')

    ps_ij = get_psij(norms_sq, 30)
    ax_psij = fig.add_subplot(2, 2, 1)
    ax_im = ax_psij.matshow(ps_ij, cmap='coolwarm', norm=LogNorm(vmin=1e-4, vmax=1e-2))

    def update_perp(perplexity):
        global ps_ij, PERPLEXITY
        PERPLEXITY = perplexity
        ps_ij = get_psij(norms_sq, PERPLEXITY)
        ax_im.set_data(ps_ij)

    # ax_psij.colorbar()
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    perp_slider = Slider(axfreq, 'Perplexity', 1, datapoints - 1, valinit=PERPLEXITY, valstep=1)
    perp_slider.on_changed(update_perp)
    ax_psij.axis('off')

    plt.subplots_adjust(bottom=0.25)

    button_onoff = Button(plt.axes([0.65, 0.025, 0.1, 0.04]), 'On/Off', hovercolor='0.975')
    pause = True
    def onoff(event):
        print(event)
        global pause
        pause = not pause
    button_onoff.on_clicked(onoff)

    button_reset = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), 'reset', hovercolor='0.975')
    def reset(event):
        global projection, calculations
        projection = np.random.normal(0, 0.0001, size=(ps_ij.shape[0], 2))
        calculations = 0
    button_reset.on_clicked(reset)

    # Now lets start with a random projection onto 2 dimensions
    projection = np.random.normal(0, 0.0001, size=(ps_ij.shape[0], 2))

    ax_qprob = fig.add_subplot(2, 2, 2)
    ax_qsij = ax_qprob.matshow(get_qsij(projection), cmap='coolwarm', norm=LogNorm(vmin=1e-4, vmax=1e-2))

    ax_sne = fig.add_subplot(2, 2, 4)
    scat = ax_sne.scatter(projection[:, 0], projection[:, 1], c=[classcols[c_id] for c_id in class_id])
    ax_sne.set_xlim(np.min(projection[:, 0]) * 1.1, np.max(projection[:, 0]) * 1.1)
    ax_sne.set_ylim(np.min(projection[:, 1]) * 1.1, np.max(projection[:, 1]) * 1.1)
    ax_sne.axis('off')

    def update(i):
        global scat, projection, ps_ij, ax_sne, pause, calculations
        if not pause:
            projection, qs_ij = gradient_descent(projection, ps_ij, 1)
            scat.set_offsets(projection)
            ax_sne.set_xlim(np.min(projection[:, 0]) * 1.1, np.max(projection[:, 0]) * 1.1)
            ax_sne.set_ylim(np.min(projection[:, 1]) * 1.1, np.max(projection[:, 1]) * 1.1)
            ax_qsij.set_data(qs_ij)


    ani = FuncAnimation(fig, update, interval=1)

    plt.show()
