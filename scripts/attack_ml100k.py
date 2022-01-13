import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluators import Kaggle, Ladder
from attackers import BoostingAttacker, KNNPosteriorBoostingAttackerDiscrete, RandomWindowSearchBoostingAttacker
from utils.data_handler import *
from scripts.create_dataset import downsample_balanced_order_preserved


def calc_uu_dist(user_dic):
    n_user = len(user_dic)
    uu_dist = np.zeros((n_user, n_user))

    for u1 in tqdm(range(n_user), desc='calc. uu_dist'):
        for u2 in range(u1, n_user):
            att1 = user_dic[u1]
            att2 = user_dic[u2]

            uu_dist[u1, u2] = \
                np.abs(att1['age'] - att2['age'])/100 \
                + (att1['sex'] != att2['sex'])*1 \
                + (att1['occupation'] != att2['occupation'])*1

    return uu_dist + uu_dist.T


def calc_ii_dist(item_dic):
    n_item = len(item_dic)
    ii_dist = np.zeros((n_item, n_item))

    for i1 in tqdm(range(n_item), desc='calc. ii_dist'):
        for i2 in range(i1, n_item):
            genres1 = np.array(item_dic[i1]['genres'])
            genres2 = np.array(item_dic[i2]['genres'])

            ii_dist[i1, i2] = np.sum(np.abs(genres1 - genres2))

    return ii_dist + ii_dist.T


if __name__ == '__main__':
    # ----- Settings ------
    # Path
    load_path_ = os.path.join('..', 'data', 'ml-100k')

    # Dataset
    part_ = 3
    file_name_ = 'u%d.test' % part_

    do_filter = False
    max_u_ = 100
    max_i_ = 150

    n_ds_ = 500  # 100

    # General
    n_rept_ = 500
    n_query_ = 100

    # ----- Load data -----
    edges_ = get_edge_list_from_file_ml100k(load_path_, file_name_, do_sort=True)

    user_dic_ = get_users_attributes_from_file_ml100k(load_path_)

    item_dic_ = get_items_attributes_from_file_ml100k(load_path_)

    # ----- Preprocess ------
    # Filter
    if do_filter:
        edges_ = [e for e in edges_ if e[0] <= max_u_ and e[1] <= max_i_]
        user_dic_ = {u: u_attr for u, u_attr in user_dic_.items() if u <= max_u_}
        item_dic_ = {i: i_attr for i, i_attr in item_dic_.items() if i <= max_i_}

    # Extract binary ratings
    y_ = np.array([(e[2] > 3)*1 for e in edges_])

    # Extract user-item pairs
    ui_ordered_list_ = [(e[0], e[1]) for e in edges_]

    # Downsample
    y_ds_, ds_indices = downsample_balanced_order_preserved(y_, n_ds_)

    ui_ordered_list_ds_ = [ui_ordered_list_[idx] for idx in ds_indices]

    # Calc. pairwise distances
    uu_dist_ = calc_uu_dist(user_dic_)

    ii_dist_ = calc_ii_dist(item_dic_)

    # ----- Choose evaluators -----
    ev_constructors_ = [
        lambda: Kaggle(y_ds_, decimals=3),
        lambda: Ladder(y_ds_, eta=0.02),
    ]
    n_ev_ = len(ev_constructors_)

    # ----- Choose attackers -----
    # Pretrain KNN MAP
    knn_map_ = KNNPosteriorBoostingAttackerDiscrete(ev_constructors_[0](), uu_dist_, ii_dist_, ui_ordered_list_ds_,
                                                    subset_size=7, k=3, exploration=0.1, conf=0.99,
                                                    compare_to_min_loss=True)
    knn_map_.fit(verbose=True)

    att_constructors_ = [
        lambda: BoostingAttacker(ev_, compare_to_min_loss=True),
        lambda: RandomWindowSearchBoostingAttacker(ev_, w=21, alpha=0.5, compare_to_min_loss=True),
        lambda: knn_map_.copy(ev_)
    ]
    n_att_ = len(att_constructors_)

    # ----- Big loop -----
    risks_ = np.zeros((n_rept_, n_ev_, n_att_, n_query_))
    for rept in tqdm(range(n_rept_), desc='attack'):
        for i_ev, ev_construct in enumerate(ev_constructors_):
            for i_att, att_construct in enumerate(att_constructors_):
                ev_ = ev_construct()
                att_ = att_construct()

                for q in range(n_query_):
                    # Query
                    att_.query_and_update()

                    # Predict
                    y_pr_ = att_.predict()

                    # Look at actual risk
                    risks_[rept, i_ev, i_att, q] = ev_.actual_risk(y_pr_)

    mean_risks_ = np.nanmean(risks_, axis=0)
    std_risks_ = np.nanstd(risks_, axis=0)

    # ----- Plotting -----
    cl = lambda idx, tot: [idx/tot, 0, (tot - 1 - idx)/tot]

    # Plot per attacker
    plt.figure(figsize=(4*n_att_, 4))
    for i_att, att_construct in enumerate(att_constructors_):
        plt.subplot(1, n_att_, i_att + 1)

        legends = []
        for i_ev, ev_construct in enumerate(ev_constructors_):
            plt.plot(range(n_query_), mean_risks_[i_ev, i_att], '-', color=cl(i_ev, n_ev_))
            plt.fill_between(range(n_query_),
                             mean_risks_[i_ev, i_att] - std_risks_[i_ev, i_att]/np.sqrt(n_rept_),
                             mean_risks_[i_ev, i_att] + std_risks_[i_ev, i_att]/np.sqrt(n_rept_),
                             color=cl(i_ev, n_ev_), alpha=0.5, label='_nolegend_')

            legends.append(ev_construct().to_string())
        plt.title(att_construct().to_string())
        plt.xlabel('#queries')
        plt.ylabel('average loss')
        plt.legend(legends, loc='upper right')

    plt.tight_layout()

    # Plot per evaluator
    plt.figure(figsize=(4*n_ev_, 4))
    for i_ev, ev_construct in enumerate(ev_constructors_):
        plt.subplot(1, n_ev_, i_ev + 1)

        legends = []
        for i_att, att_construct in enumerate(att_constructors_):
            plt.plot(range(n_query_), mean_risks_[i_ev, i_att], '-', color=cl(i_att, n_att_))
            plt.fill_between(range(n_query_),
                             mean_risks_[i_ev, i_att] - std_risks_[i_ev, i_att] / np.sqrt(n_rept_),
                             mean_risks_[i_ev, i_att] + std_risks_[i_ev, i_att] / np.sqrt(n_rept_),
                             color=cl(i_att, n_att_), alpha=0.5, label='_nolegend_')

            legends.append(att_construct().to_string())
        plt.title(ev_construct().to_string())
        plt.xlabel('#queries')
        plt.ylabel('average loss')
        plt.legend(legends, loc='upper right')

        plt.ylim((np.min(mean_risks_), np.max(mean_risks_)))

    plt.tight_layout()
