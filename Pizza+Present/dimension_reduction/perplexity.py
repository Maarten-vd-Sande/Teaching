import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def estimate_sigmas(norms_sq, target_perplexity, tlower=1e-10, tupper=1e10, tol=1e-4, max_iter=1000):
    lower, upper = tlower, tupper
    for i in range(max_iter):
        # guess a sigma value
        sigma = (lower + upper) / 2

        # calculate the corresponding conditional p_ji val
        p_ji = np.exp(-norms_sq / (sigma * sigma))
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
    return sigma, p_ji


# random distances
distances = np.random.uniform(0, 1, size=15)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(bottom=0.25)

sigma = 3
xs = np.arange(0, 1, 1/100)
ys = np.exp(-xs**2 / 2*sigma**2)

normal = ax.plot(xs, ys)[0]
plt.subplots_adjust(bottom=0.25)
scats = ax.scatter(distances, np.zeros(15) - 0.1, s=100*np.exp(-distances**2 / 2*sigma**2))


def update(perplexity):
    global scats, normal
    sigma, p_ji = estimate_sigmas(distances**2, perplexity)

    normal.set_data(xs, np.exp(-xs**2 / (2*sigma**2)))
    scats.set_sizes(1000*p_ji)


axdim = plt.axes([0.25, 0.1, 0.65, 0.03])
axdim = Slider(axdim, 'perplexity', 1, 15, valinit=3, valstep=0.5)
axdim.on_changed(update)
update(3)
ax.axis('off')

plt.show()
