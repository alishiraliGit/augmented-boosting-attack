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
    }

    explorations = [1, 0.1]
    n_exp = len(explorations)

    # General
    n_run = 100
    n_rept = 5
    n_query = 100

    assert n_run <= data_sett['n_run']

    # ----- Load data -----
    print('Loading data ...')

    Y_te_list = []
    for i_e, exploration in enumerate(explorations):
        data_sett['exploration'] = exploration

        data_dic = load_knnrecsys_synthetic_data(load_path, **data_sett)

        Y_te_list.append(data_dic['data']['ds_labels'][:n_run])

    # ----- Choose evaluators -----
    eval_constructors = [
        lambda y: Kaggle(y, decimals=5),
        lambda y: Kaggle(y, decimals=2),
        lambda y: Ladder(y, eta=0.001),
        lambda y: Ladder(y, eta=0.02),
    ]
    n_ev = len(eval_constructors)

    # ----- Choose attackers -----
    attacker_constructor = \
        lambda e: AdaptiveRandomWindowSearchBoostingAttacker(e, w=21, alpha=0.5, compare_to_min_loss=True)

    # ----- Big loop -----
    print('Attacking ...')

    risks = np.zeros((n_rept, n_run, n_ev, n_exp, n_query))
    for run in tqdm(range(n_run)):
        for rept in range(n_rept):
            for i_exp, exploration in enumerate(explorations):
                y_te = Y_te_list[i_exp][run]

                for i_ev, ev_construct in enumerate(eval_constructors):
                    ev = ev_construct(y_te)
                    att = attacker_constructor(ev)

                    for q in range(n_query):
                        # Query
                        att.query_and_update()

                        # Predict
                        y_pr = att.predict()

                        # Look at actual risk
                        risks[rept, run, i_ev, i_exp, q] = ev.actual_risk(y_pr)

    mean_risks = np.nanmean(risks, axis=(0, 1))
    std_risks = np.nanstd(risks, axis=(0, 1))

    # ----- Plotting -----
    cl = lambda idx, tot: [idx/tot, 0, (tot - 1 - idx)/tot]

    # Plot per exploration
    plt.figure(figsize=(4*n_exp, 4))
    for i_exp, exploration in enumerate(explorations):
        plt.subplot(1, n_exp, i_exp + 1)

        legends = []
        for i_ev, ev_construct in enumerate(eval_constructors):
            plt.plot(range(n_query), mean_risks[i_ev, i_exp], '-', color=cl(i_ev, n_ev))
            plt.fill_between(range(n_query),
                             mean_risks[i_ev, i_exp] - std_risks[i_ev, i_exp]/np.sqrt(n_run*n_rept),
                             mean_risks[i_ev, i_exp] + std_risks[i_ev, i_exp]/np.sqrt(n_run*n_rept),
                             color=cl(i_ev, n_ev), alpha=0.5, label='_nolegend_')

            legends.append(ev_construct([1]).to_string())
        plt.title('exploration=%.1f' % exploration)
        plt.xlabel('#queries')
        plt.ylabel('average loss')
        plt.legend(legends, loc='upper right')

    plt.tight_layout()

    # Plot per evaluator
    plt.figure(figsize=(4*n_ev, 4))
    for i_ev, ev_construct in enumerate(eval_constructors):
        plt.subplot(1, n_ev, i_ev + 1)

        legends = []
        for i_exp, exploration in enumerate(explorations):
            plt.plot(range(n_query), mean_risks[i_ev, i_exp], '-', color=cl(i_exp, n_exp))
            plt.fill_between(range(n_query),
                             mean_risks[i_ev, i_exp] - std_risks[i_ev, i_exp] / np.sqrt(n_run * n_rept),
                             mean_risks[i_ev, i_exp] + std_risks[i_ev, i_exp] / np.sqrt(n_run * n_rept),
                             color=cl(i_exp, n_exp), alpha=0.5, label='_nolegend_')

            legends.append('exploration=%.1f' % exploration)
        plt.title(ev_construct([1]).to_string())
        plt.xlabel('#queries')
        plt.ylabel('average loss')
        plt.legend(legends, loc='upper right')

        plt.ylim((np.min(mean_risks), np.max(mean_risks)))

    plt.tight_layout()
