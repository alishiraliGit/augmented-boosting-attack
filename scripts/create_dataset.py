import os
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from simulators import KNNRecSys
from demo_knnrecsys import plot_regions
from utils.data_handler import stringify

rng = np.random.default_rng(2)


def downsample_balanced_order_preserved(y, n_ds):
    n_0 = np.sum((y == 0)*1)
    n_1 = np.sum((y == 1)*1)

    n = np.min([n_0, n_1, n_ds])

    indices_1 = rng.choice(range(len(y)), size=n, replace=False, p=(y == 1)/n_1)
    indices_0 = rng.choice(range(len(y)), size=n, replace=False, p=(y == 0)/n_0)

    indices = np.concatenate((indices_0, indices_1))
    sorted_indices = np.sort(indices)

    return y[sorted_indices], sorted_indices


if __name__ == '__main__':
    # ----- Settings -----
    # Path
    save_path = os.path.join('..', 'data')

    # Simulator
    simul_sett = {
        'n_c_star': 300,
        'k_star': 1,
        'n_sample': 2000,
        'k': 1,
        'exploration': 0.1,
    }

    # Dataset
    data_sett = {
        'n_run': 100,
        'n_ds': 500,
    }

    # ----- Instantiate a simulator -----
    simul = KNNRecSys(n_c_star=simul_sett['n_c_star'], k_star=simul_sett['k_star'])

    # ----- Plot ground truth -----
    plt.figure(figsize=(4, 4))
    plot_regions(simul.centers_star, simul.labels_star, simul.k_star)
    plt.xlabel('user')
    plt.ylabel('item')
    plt.tight_layout()

    # ----- Loop -----
    all_centers = np.zeros((data_sett['n_run'], simul_sett['n_sample'], 2))
    all_labels = np.zeros((data_sett['n_run'], simul_sett['n_sample']))
    all_ds_centers = np.zeros((data_sett['n_run'], 2*data_sett['n_ds'], 2))
    all_ds_labels = np.zeros((data_sett['n_run'], 2*data_sett['n_ds']))

    for run in tqdm(range(data_sett['n_run'])):
        # Run simulator
        centers, labels = simul.run(n_sample=simul_sett['n_sample'],
                                    k=simul_sett['k'],
                                    exploration=simul_sett['exploration'])

        # Downsampling
        ds_labels, ds_indices = downsample_balanced_order_preserved(labels, data_sett['n_ds'])
        ds_centers = centers[ds_indices]

        # Add to dataset
        all_centers[run] = centers
        all_labels[run] = labels
        all_ds_centers[run] = ds_centers
        all_ds_labels[run] = ds_labels

    # ----- Save dataset -----
    all_sett = simul_sett.copy()
    all_sett.update(data_sett)
    file_name = 'knnrecsys%s.pkl' % stringify(all_sett)

    with open(os.path.join(save_path, file_name), 'wb') as f:
        pickle.dump({
            'data': {
                'centers': all_centers,
                'labels': all_labels,
                'ds_centers': all_ds_centers,
                'ds_labels': all_ds_labels
            },
            'settings': {
                'simulator': simul_sett,
                'dataset': data_sett,
            }
        }, f)
