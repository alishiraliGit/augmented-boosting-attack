import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from utils.data_handler import load_knnrecsys_synthetic_data
from evaluators import Kaggle, Ladder
from attackers import BoostingAttacker, CorrelatedBoostingAttacker,\
    AdaptiveRandomWindowSearchBoostingAttacker, KNNMAPBoostingAttacker

if __name__ == '__main__':
    # ----- Settings -----
    # Data
    load_path = os.path.join('..', 'data')
    data_sett = {
        'n_run': 100,
        'n_sample': 1000,
        'n_ds': 200,
        'n_c_star': 100,
        'k_star': 1,
        'k': 1,
        'exploration': 0.1,
    }

    # General
    n_run = 30
    n_rept = 5
    n_query = 100

    assert n_run <= data_sett['n_run']

    # ----- Load data -----
    print('Loading data ...')

    data_dic = load_knnrecsys_synthetic_data(load_path, **data_sett)

    Y_te = data_dic['data']['ds_labels'][:n_run]

    # ----- Choose evaluators -----
    eval_constructors = [
        lambda y: Kaggle(y, decimals=3),
        lambda y: Ladder(y, eta=0.02),
    ]
    n_ev = len(eval_constructors)

    # ----- Choose attackers -----
    attacker_constructors = [
        lambda e: BoostingAttacker(e, compare_to_min_loss=True),
        # lambda e: CorrelatedBoostingAttacker(e, gamma=0.2, compare_to_min_loss=True),
        lambda e: AdaptiveRandomWindowSearchBoostingAttacker(e, w=21, alpha=0.5, compare_to_min_loss=True),
        # lambda e: KNNMAPBoostingAttacker(e, centers=data_dic['data']['ds_centers'][run],
        #                                 k=1, N=4, exploration=0.1, conf=0.99,
        #                                 do_grouping=True, grouping_depth=1,
        #                                 compare_to_min_loss=True,
        #                                 verbose=False)
    ]
    n_att = len(attacker_constructors)

    # ----- Big loop -----
    print('Attacking ...')

    risks = np.zeros((n_rept, n_run, n_ev, n_att, n_query))
    for run in tqdm(range(n_run)):
        for rept in range(n_rept):
            y_te = Y_te[run]

            for i_ev, ev_construct in enumerate(eval_constructors):
                for i_att, att_construct in enumerate(attacker_constructors):
                    ev = ev_construct(y_te)
                    att = att_construct(ev)

                    for q in range(n_query):
                        # Query
                        att.query_and_update()

                        # Predict
                        y_pr = att.predict()

                        # Look at actual risk
                        risks[rept, run, i_ev, i_att, q] = ev.actual_risk(y_pr)

    mean_risks = np.nanmean(risks, axis=(0, 1))
    std_risks = np.nanstd(risks, axis=(0, 1))

    # ----- Plotting -----
    cl = lambda idx, tot: [idx/tot, 0, (tot - 1 - idx)/tot]

    # Plot per attacker
    plt.figure(figsize=(4*n_att, 4))
    for i_att, att_construct in enumerate(attacker_constructors):
        plt.subplot(1, n_att, i_att + 1)

        legends = []
        for i_ev, ev_construct in enumerate(eval_constructors):
            plt.plot(range(n_query), mean_risks[i_ev, i_att], '-', color=cl(i_ev, n_ev))
            plt.fill_between(range(n_query),
                             mean_risks[i_ev, i_att] - std_risks[i_ev, i_att]/np.sqrt(n_run*n_rept),
                             mean_risks[i_ev, i_att] + std_risks[i_ev, i_att]/np.sqrt(n_run*n_rept),
                             color=cl(i_ev, n_ev), alpha=0.5, label='_nolegend_')

            legends.append(ev_construct([1]).to_string())
        plt.title(att_construct(eval_constructors[0]([1])).to_string())
        plt.xlabel('#queries')
        plt.ylabel('average loss')
        plt.legend(legends, loc='upper right')

    plt.tight_layout()

    # Plot per evaluator
    plt.figure(figsize=(4*n_ev, 4))
    for i_ev, ev_construct in enumerate(eval_constructors):
        plt.subplot(1, n_ev, i_ev + 1)

        legends = []
        for i_att, att_construct in enumerate(attacker_constructors):
            plt.plot(range(n_query), mean_risks[i_ev, i_att], '-', color=cl(i_att, n_att))
            plt.fill_between(range(n_query),
                             mean_risks[i_ev, i_att] - std_risks[i_ev, i_att] / np.sqrt(n_run * n_rept),
                             mean_risks[i_ev, i_att] + std_risks[i_ev, i_att] / np.sqrt(n_run * n_rept),
                             color=cl(i_att, n_att), alpha=0.5, label='_nolegend_')

            legends.append(att_construct(ev_construct([1])).to_string())
        plt.title(ev_construct([1]).to_string())
        plt.xlabel('#queries')
        plt.ylabel('average loss')
        plt.legend(legends, loc='upper right')

        plt.ylim((np.min(mean_risks), np.max(mean_risks)))

    plt.tight_layout()
