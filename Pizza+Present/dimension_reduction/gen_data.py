import math
import numpy as np
import pandas as pd


# colors per class
classcols = {0.0: 'C0', 1.0: 'C1', 2.0: 'C2', 3.0: 'C3'}


def generate_mikado(nr_points):
    """

    """
    assert nr_points % 3 == 0
    class_size = nr_points // 3

    def rotate(origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    data = np.hstack((np.random.random(size=(nr_points, 1)), np.zeros((nr_points, 3))))
    for dim in range(0, 3):
        indices = np.arange(dim*class_size, (dim+1)*class_size)
        data_minx, data_maxx = np.min(data[indices, 0]), np.max(data[indices, 0])

        xy_angle = math.radians(120 * dim)
        xz_angle = math.radians(120)
        origin = (data_maxx - data_minx) / 2, (data_maxx - data_minx) / 4
        for i, point in enumerate(data[indices]):
            data[indices[i], [0, 2]] = rotate((0, 0), point[[0, 2]], xz_angle)
            data[indices[i], [0, 1]] = rotate(origin, point[[0, 1]], xy_angle)
        data[indices, -1] = dim + 1

    data[:, :3] += np.random.normal(0, 0.1, (data.shape[0], 3))
    pd_data = pd.DataFrame(data)
    pd_data.columns = ['var 1', 'var 2', 'var 3', 'class']

    return pd_data


def generate_gaussians(nr_points):
    """

    """
    assert nr_points % 3 == 0
    class_size = nr_points // 3

    data = np.zeros((nr_points, 4))
    locs = [np.array([0, 0, 0]), np.array([3, 3, 0]), np.array([12, 0, 3])]
    vars = [0.5, 0.5, 1.5]
    for i, (loc, var) in enumerate(zip(locs, vars)):
        data[i*class_size:(i+1)*class_size, :3] = loc + np.random.normal(0, var, (class_size, 3))
        data[i*class_size:(i+1)*class_size, 3] = i

    pd_data = pd.DataFrame(data)
    pd_data.columns = ['var 1', 'var 2', 'var 3', 'class']

    return pd_data
