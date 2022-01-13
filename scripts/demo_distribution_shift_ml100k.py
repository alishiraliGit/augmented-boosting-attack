import numpy as np
from matplotlib import pyplot as plt

from utils.data_handler import *


if __name__ == '__main__':
    # ----- Settings ------
    # Path
    load_path_ = os.path.join('..', 'data', 'ml-100k')

    # Dataset
    part_ = 3
    file_name_ = 'u%d.base' % part_

    # General
    n_rept_ = 500
    n_query_ = 100

    # ----- Load data -----
    edges_ = get_edge_list_from_file_ml100k(load_path_, file_name_, do_sort=True)

    # ----- Big loop ------
    user_n_obs_dic_ = {}
    r_n_obs_dic = {}
    max_n_obs_ = 0
    for (u_, i_, r_, t_) in edges_:
        if u_ in user_n_obs_dic_:
            user_n_obs_dic_[u_] += 1
        else:
            user_n_obs_dic_[u_] = 0

        if user_n_obs_dic_[u_] in r_n_obs_dic:
            r_n_obs_dic[user_n_obs_dic_[u_]].append(r_)
        else:
            r_n_obs_dic[user_n_obs_dic_[u_]] = [r_]
            max_n_obs_ = user_n_obs_dic_[u_]

    r_mean_ = np.zeros((max_n_obs_,))

    for n_obs_ in range(max_n_obs_):
        r_mean_[n_obs_] = np.mean(r_n_obs_dic[n_obs_])

    plt.figure(figsize=(4, 4))
    plt.plot(range(max_n_obs_), r_mean_)
    # plt.xlim((0, 20))
