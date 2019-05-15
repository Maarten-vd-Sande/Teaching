import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.widgets import Button


if __name__ == "__main__":
    # generate some random data
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=2.5, random_state=42)

    # we 'expect' 5 clusters (k)
    k = 5
    # random start centers and non-assigned group
    centroids = X[np.random.choice(np.arange(X.shape[0]), k, replace=False)]
    groups = k * np.ones(X.shape[0], dtype=int)

    # setup the figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax_data = ax.scatter(X[:, 0], X[:, 1], c=[f"C{g}" for g in groups])
    ax_centroids = ax.scatter(centroids[:, 0], centroids[:, 1], c='k', s=250, marker='*')
    plt.axis('off')

    # make the k-means buttons
    def update_centroids(event):
        """ set the centroids in the middle of the dots that are in its group """
        global centroids, ax_centroids, centroids, X, groups
        centroids = np.array([X[groups == k].mean(axis=0) for k in range(k)])
        ax_centroids.set_offsets(centroids)
        plt.draw()

    button_centroids = Button(plt.axes([0.45, 0.025, 0.2, 0.04]), 'centroid to center', hovercolor='0.975')
    button_centroids.on_clicked(update_centroids)

    def random_centroids(event):
        """ random initial state """
        global centroids, ax_centroids, centroids, X, groups
        centroids = X[np.random.choice(np.arange(X.shape[0]), k, replace=False)]
        ax_centroids.set_offsets(centroids)
        groups = k * np.ones(X.shape[0], dtype=int)
        ax_data.set_color([f'C{g}' for g in groups])
        plt.draw()
    button_rand_centroids = Button(plt.axes([0.15, 0.025, 0.2, 0.04]), 'random centroids', hovercolor='0.975')
    button_rand_centroids.on_clicked(random_centroids)

    def update_points(event):
        """ appoint each sample to the closest centroid """
        global centroids, ax_centroids, centroids, X, groups
        distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
        groups = np.argmin(distances, axis=0)
        ax_data.set_color([f'C{g}' for g in groups])
        plt.draw()
    button_closest = Button(plt.axes([0.75, 0.025, 0.2, 0.04]), 'group on centroid', hovercolor='0.975')
    button_closest.on_clicked(update_points)

    ax.axis('equal')
    plt.show()
