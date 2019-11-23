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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--classifier', type=str, default='FeatureBasedClassifier',
        help='Classifier can be either FeatureBasedClassifier or SpectralClassifier')
    parser.add_argument(
        '--nsamples', type=int, default=10)
    parser.add_argument(
        '--dataset', type=str, default='T2001-06',
        help='Dataset can be either T2001-06 or CHSIM')
    parser.add_argument(
        '--dim', type=int, default=10,
        help='Dimension for spectral classifier. Required only if classifier is set to SpectralClassifier.')
    parser.add_argument(
        '--emb-type', type=str, default='dual',
        help='Type of spectral embeddings to compute. It can be either primal or dual')
    parser.add_argument(
        '--pool', type=str, default='max',
        help='Type of pooling to be used for spectral embeddings. Required only if classifier is set to SpectralClassifier and emb-type is set to primal.')
    
    args = parser.parse_args()

    if args.dataset == 'T2001-06':
        perpetrators, terror_plots = reader.read_ty2001_06_dataset()
        hyperedges, hyperedges_timestamps, total_nodes = (
            reader.get_hyperedges_from_ty2001_06_dataset(
                perpetrators, terror_plots))
        ground, train, test = (
            utils.split_hyperedges_according_to_time_for_ty2001_06_dataset(
                hyperedges, hyperedges_timestamps))
        node_active = None
    else:
        hyperedges, node_active = (
            reader.read_chsim_dataset())
        total_nodes = len(node_active)
        ground, train, test = utils.split_chsim_dataset(hyperedges)

    Htest = utils.hyperedges_to_incidence_matrix(test, total_nodes)
    
    if args.classifier == 'FeatureBasedClassifier':
        print('Executing feature based classifier for hyperedge prediction')
        features = list(featurizer.FEATURE_MAP.keys())

        clfs = []
        roc_scores = []
        for i in range(args.nsamples):
            roc_score, clf = predictor.train_feature_based_classifier(
                ground, train, test, features, total_nodes, node_active)
            clfs.append(clf)
            roc_scores.append(roc_score)
        if args.nsamples > 1:
            coefs = np.array(
                [clf.coef_[0].tolist() for clf in clfs]).mean(axis=0).tolist()
        else:
            coefs = clfs[0].coef_[0].tolist()
        feature_weights = {
            feat: weight for feat, weight in zip(features, coefs)}
        important_features = filter(
            lambda x: x[1] != 0,
            sorted(feature_weights.items(), key=lambda x: x[1], reverse=True))
        print('Important Features for Hyperedge Prediction are:')
        tab = PrettyTable()
        tab.field_names = ['Feature', 'Weight']
        for feat, weight in important_features:
            tab.add_row([feat, weight])
        print(tab)

        print('\n\n Average ROC Score for prediction: %.6f' % (
            np.mean(roc_scores)))

    else:
        print('Executing spectral classifier for hyperedge prediction')

        clfs = []
        roc_scores = []
        for i in range(args.nsamples):
            roc_score, clf = predictor.train_spectral_classifier(
                ground, train, test, total_nodes, args.dim, args.emb_type,
                args.pool, node_active)
            clfs.append(clf)
            roc_scores.append(roc_score)

        print('\n\n Average ROC Score for prediction: %.6f' % (
            np.mean(roc_scores)))
