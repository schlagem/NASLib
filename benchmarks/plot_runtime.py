#%load_ext autoreload
#%autoreload 2'
#%matplotlib inline

import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter

def get_results(predictor, constraint, path, epochs, metric='valid_acc', dataset='cifar10', ug=False):

    result_file = os.path.join(path, predictor, constraint[:-1], constraint[-1], 'errors.json')
    result = json.load(open(result_file))

    config = result[0]
    val_acc = result[1]['valid_acc'][:epochs]
    val_err = [100 - x for x in val_acc]

    surr_time = np.array(result[1]['runtime'])[:epochs]
    if ug:
        runtime = 200 * np.array(result[1]['runtime'])[:epochs] + surr_time #'train_time'
    else:
        runtime = np.array(result[1]['runtime'])[:epochs] + surr_time #'train_time'

    incumbent = [min(val_err[:epoch]) for epoch in range(1, len(val_err) + 1)]
    runtime = [sum(runtime[:epoch]) for epoch in range(1, len(runtime) + 1)]

    output = np.array(incumbent)
    time = np.array(runtime)

    print(predictor, constraint, 'output shape', output.shape)
    return output, time

if __name__ == "__main__":
    # set up colors and plot markings
    defaults = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]

    # goes up to 24
    c_max = 9
    colors = [*defaults[:c_max], *defaults[:c_max], *defaults[:c_max]]
    fmts = [*['-'] * c_max, *['--'] * c_max, *[':'] * c_max]
    markers = [*['^'] * c_max, *['v'] * c_max, *['o'] * c_max]

    # https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html


    pred_label_dict = {
        'valloss': 'Val. Loss', 'valacc': 'Val. Acc.', 'sotl': 'SoTL', 'bananas': 'BANANAS',
        'mlp': 'Feedforward', 'gbdt': 'GBDT', 'gcn': 'GCN', 'bonas_gcn': 'BONAS', 'xgb': 'XGB',
        'ngb': 'NGB', 'rf': 'RF', 'jacov': 'Jacob. Cov.', 'dngo': 'DNGO', 'bohamiann': 'BOHAMIANN',
        'bayes_lin_reg': 'Bayes. Lin. Reg.', 'ff_keras': 'FF-Keras', 'gp': 'GP', 'sparse_gp': 'Sparse GP',
        'var_sparse_gp': 'Var. Sparse GP', 'seminas': 'SemiNAS', 'lcsvr': 'LcSVR', 'snip': 'SNIP', 'sotle': 'SoTLE',
        'bonas': 'BONAS', 'omni_lofi': 'Omni Lofi', 'nao': 'NAO', 'lgb': 'LGB', 'none': 'True', 'pretrained': 'pretrained'
    }

    # set up parameters for the experiments
    epochs = 300
    results_dict = {}

    folder = os.path.expanduser('docs\\rs_run\\imagenet\\runs')
    predictors = ['pretrained'] # 'mlp', 'lgb', 'xgb', 'rf', 'bayes_lin_reg', 'gp', 'pretrained'
    constraints = ['latency1', 'latency2', 'latency3', 'none0', 'parameters1', 'parameters2', 'parameters3'] # 'latency1', 'latency2', 'latency3', 'none0', 'parameters1', 'parameters2', 'parameters3'
    for i, predictor in enumerate(predictors):
        for constraint in constraints:
            output, runtime = get_results(predictor, constraint, folder, epochs=epochs, metric='valid_acc', ug=False) #'test_acc' #mean, std, std_error,
            label = pred_label_dict[predictor] + ' ' + constraint[:-1] + ' ' + constraint[-1]
            results_dict[predictor + constraint] = {'label': label,
                                       'key': predictor + constraint, 'mean': output, 'runtime': runtime #'std': std,
                                       } #'std_error': std_error, 'runtime': runtime

    ### plot performance vs runtime ###
    # didn't run them long enough to do logspace here. (These experiments took surprisingly long to run)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = 'dotted'

    fig, ax = plt.subplots(figsize=[8, 4])
    for i, key in enumerate(predictors):
        for j, constraint in enumerate(constraints):
            key_with_constraint = key + constraint
            y = results_dict[key_with_constraint]['mean']
            label = results_dict[key_with_constraint]['label']
            x = results_dict[key_with_constraint]['runtime']

            color_index = i * len(constraints) + j

            ax.plot(x, y, label=label, color=colors[color_index], linestyle=fmts[color_index])

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xscale('log')
    # ax.set_ylim([7, 11])
    # ax.set_xlim([1e4, 1.6e6])

    ax.legend(loc=(1.04, 0))
    ax.set_xlabel('Runtime [s]')
    ax.set_ylabel('Validation error (%)')
    ax.grid(True, which="both", ls="-", alpha=.5)
    ax.set_title('Validation error vs. search time with RS as optimizer')
    plt.savefig('test_single_lines.pdf', bbox_inches='tight', pad_inches=0.1)
