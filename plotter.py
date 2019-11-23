import utils
import hyperedge_features as featurizer
import negative_sampler as sampler
import reader
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
from prettytable import PrettyTable
import predictor
import matplotlib.pyplot as plt

# hyperedges, total_nodes = reader.read_chsim_dataset()
# ground, train, test = utils.split_chsim_dataset(hyperedges)

perpetrators, terror_plots = reader.read_ty2001_06_dataset()
hyperedges, hyperedges_timestamps, total_nodes = (
    reader.get_hyperedges_from_ty2001_06_dataset(
        perpetrators, terror_plots))
ground, train, test = (
    utils.split_hyperedges_according_to_time_for_ty2001_06_dataset(
        hyperedges, hyperedges_timestamps))
node_active = None

for pool in ['max', 'avg']:
    for dim in [10, 20, 30, 40, 50]:
        roc_score, clf, curve = predictor.train_spectral_classifier(
            ground, train, test, total_nodes, dim, 'primal',
            pool)
        plt.plot(curve[0], curve[1], label='TA2001-16-%d-DIM' % dim)

    plt.plot(
        [i for i in np.arange(0, 1, 0.1)], [i for i in np.arange(0, 1, 0.1)],
        color='gray', linestyle='--', alpha=0.3)
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Positve Rate')
    plt.title('TA2001-16 Primal %s-pool ROC Curve' % pool.title())
    plt.legend()
    plt.rcParams['figure.figsize'] = 10, 8
    plt.show()

for dim in [10, 20, 30, 40, 50]:
    roc_score, clf, curve = predictor.train_spectral_classifier(
        ground, train, test, total_nodes, dim, 'dual',
        pool)
    plt.plot(curve[0], curve[1], label='TA2001-16-%d-DIM' % dim)

plt.plot(
    [i for i in np.arange(0, 1, 0.1)], [i for i in np.arange(0, 1, 0.1)],
    color='gray', linestyle='--', alpha=0.3)
plt.xlabel('False Positve Rate')
plt.ylabel('True Positve Rate')
plt.title('TA2001-16 Dual ROC Curve')
plt.legend()
plt.rcParams['figure.figsize'] = 10, 8
plt.show()

# features = list(featurizer.FEATURE_MAP.keys())
# for i in range(5):
#     roc_score, clf, curve = predictor.train_feature_based_classifier(
#         ground, train, test, features, total_nodes, node_active)
#     plt.plot(curve[0], curve[1])
# plt.xlabel('False Positve Rate')
# plt.ylabel('True Positve Rate')
# plt.title('TY2001-06 ROC Curve')
# plt.legend()
# plt.rcParams['figure.figsize'] = 10, 8
# plt.show()