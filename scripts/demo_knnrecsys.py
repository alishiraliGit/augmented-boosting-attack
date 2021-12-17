import os

import numpy as np
from matplotlib import pyplot as plt

from simulators import KNNRecSys


def plot_regions(centers, labels, k, eps=1e-2):
    # Init. a mesh
    users = np.arange(0, 1, eps)
    items = np.arange(0, 1, eps)
    n_sample = len(users)*len(items)

    U, I = np.meshgrid(users, items)
    U_flat, I_flat = np.ravel(U), np.ravel(I)

    # Calc. function for mesh
    R_flat = np.zeros((n_sample,))

    for idx in range(n_sample):
        u = U_flat[idx]
        i = I_flat[idx]

        R_flat[idx] = KNNRecSys.score(u, i, centers=centers, labels=labels, k=k)

    plt.scatter(U_flat, I_flat, c=R_flat, cmap='bwr')
    plt.plot(centers[labels == 0, 0], centers[labels == 0, 1], 'ko')
    plt.plot(centers[labels == 1, 0], centers[labels == 1, 1], 'k^')


if __name__ == '__main__':
    # ----- Settings -----
    sett = {
        'n_c_star': 100,
        'k_star': 1,
        'n_sample': 1000,
        'k': 1,
    }

    # ----- Instantiate a simulator ------
    simulator = KNNRecSys(n_c_star=sett['n_c_star'], k_star=sett['k_star'])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plot_regions(simulator.centers_star, simulator.labels_star, simulator.k_star)
    plt.xlabel('user')
    plt.ylabel('item')
    plt.title('Ground Truth')

    # ----- Run 1 (low exploration) -----
    cents, ls = simulator.run(n_sample=sett['n_sample'], k=sett['k'], exploration=0.1)

    plt.subplot(1, 3, 2)
    plot_regions(cents, ls, 1)
    plt.xlabel('user')
    plt.ylabel('item')
    plt.title('Collected Samples (low exploration)')

    # ----- Run 2 (high exploration) -----
    cents, ls = simulator.run(n_sample=sett['n_sample'], k=sett['k'], exploration=1)

    plt.subplot(1, 3, 3)
    plot_regions(cents, ls, 1)
    plt.xlabel('user')
    plt.ylabel('item')
    plt.title('Collected Samples (high exploration)')

    plt.tight_layout()

    plt.savefig(os.path.join('..', 'results', 'figs', 'demo_recsys.pdf'))

