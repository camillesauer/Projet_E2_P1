import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def viz_model(search):
    test_scores = search.cv_results_['mean_test_score']
    train_scores = search.cv_results_['mean_train_score']

    plt.figure(figsize=(15, 9))
    plt.title(label="Validation Curves",
              fontsize=18,
              color="black",
              fontweight=20,
              pad='2.0', y=1.06)
    plt.plot(test_scores, label='Test')
    plt.plot(train_scores, label='Train')
    plt.legend(loc='best')
    plt.ylabel("Mean_test_score", labelpad=15, color='black')
    plt.xlabel("Mean_train_score", labelpad=15, color='black')
    plt.show()

    return plt

def viz_params(search):
    results = search.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']
    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(search.best_params_.keys())
    for p_k, p_v in search.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))

    params = search.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle('Score per parameter', fontsize=20)
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^', label='train')
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
    return fig