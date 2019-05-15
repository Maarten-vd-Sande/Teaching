#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import gen_data


def ondraw(event):
    # get the projection
    xs_proj, ys_proj, zs_proj, *_ = proj3d.proj_transform(xs, ys, zs, ax3D.get_proj())
    xyz_proj = np.array([xs_proj, ys_proj, zs_proj]).T

    # calculate the variance per axis
    vars_proj = xyz_proj.var(axis=0)
    vars_norm = vars_proj / np.sum(vars_proj)

    # update bars
    for i, rect in enumerate(rects):
        rect.set_height(vars_norm[i])


if __name__ == "__main__":
    # generate some random 3D data
    xs, ys, zs = np.random.multivariate_normal([0, 0, 0], [[1, 2, 3],
                                                           [2, 1, 2],
                                                           [3, 2, 1]], size=250).T

    # setup the figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle('Manual PCA')

    # setup the 3D axis
    ax3D = fig.add_subplot(1, 2, 1, projection='3d')
    ax3D.scatter(xs, ys, zs)
    cid = fig.canvas.mpl_connect('draw_event', ondraw)

    # setup the variance captured barplot
    ax2D = fig.add_subplot(1, 2, 2)
    rects = ax2D.bar(range(3), [0, 0, 0], align='center', alpha=0.5)
    ax2D.grid(False); ax2D.set_ylim([0, 1])
    ax2D.set_xticks([0, 1, 2]); ax2D.set_xticklabels(['Manual axis 1', 'Manual axis 2', 'Reduced axis'])
    ax2D.set_ylabel('Explained variance')

    ondraw(None)
    plt.show()
