import utils
import hyperedge_features as featurizer
import negative_sampler as sampler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
from prettytable import PrettyTable

def read_dataset():
    '''Read dataset from file.
    
    Returns:
    DataFrame. Pandas dataframe containing data from file.
    '''
    perpetrators = pd.read_csv("./data/names_of_perpetrators.csv")
    terror_plots = pd.read_csv("./data/terror_plots.csv")
    return perpetrators, terror_plots

def get_hyperedges_from_dataset(perpetrators, terror_plots):
    '''Create a list of hyperedges from the data. Each hyperedge is a frozenset
    of the nodes it contains.

    Returns:
    hyperedges: List. List of all hyperedges
    hyperedges_timestamps: List. Timestamp associated with each hyperedge.
        Timestamps should be in same order as that of hyperedges.
    total_nodes: Int. Total number of nodes in hypergraph.
    '''
    plots = set(
        perpetrators['terror_plot'].values.tolist() +
        perpetrators['terror_plot_2'].dropna().values.tolist())
    plot_to_id = {p: i for i, p in enumerate(plots)}

    plot_with_perpetrator = (
        perpetrators[['person_ID', 'terror_plot']].dropna().values.tolist() +
        perpetrators[['person_ID', 'terror_plot_2']].dropna().values.tolist())
    
    plot_to_perpetrator = defaultdict(set)
    for pid, plot in plot_with_perpetrator:
        plot_to_perpetrator[plot_to_id[plot]].add(pid)
    
    states = (
        perpetrators['last_residency_state'].dropna().values.tolist() +
        perpetrators['state_charged'].dropna().values.tolist())
    state_to_id = {s: i for i, s in enumerate(set(states))}
    
    perpetrator_with_res_state = perpetrators[
        ['person_ID', 'last_residency_state']].dropna().values.tolist()
    res_state_to_perpetrator = defaultdict(set)
    for pid, state in perpetrator_with_res_state:
        res_state_to_perpetrator[state_to_id[state]].add(pid)

    # perpetrator_with_charged_state = perpetrators[
    #     ['person_ID', 'state_charged']].dropna().values.tolist()
    # charged_state_to_perpetrator = defaultdict(set)
    # for pid, state in perpetrator_with_charged_state:
    #     charged_state_to_perpetrator[state_to_id[state]].add(pid)    

    plots = list(plot_to_perpetrator.keys())
    hyperedges = []
    for k in plots:
        hyperedges.append(set(plot_to_perpetrator[k]))
    for k, v in res_state_to_perpetrator.items():
        hyperedges.append(set(v))
    # for k, v in charged_state_to_perpetrator.items():
    #     hyperedges.append(set(v))
    
    plots_with_year = terror_plots[['name', 'year']].fillna(0).values.tolist()
    plot_to_year = {}
    for plot, year in plots_with_year:
        if plot in plot_to_id:
            plot_to_year[plot_to_id[plot]] = year
    
    hyperedges_timestamps = []
    for k in plots:
        hyperedges_timestamps.append(plot_to_year.get(k, 0))

    for i in range(len(res_state_to_perpetrator)):
        hyperedges_timestamps.append(0)
    
    total_nodes = reduce(lambda x, y: x.union(y), hyperedges)
    node_to_id = {node: idx for idx, node in enumerate(total_nodes)}
    for idx in range(len(hyperedges)):
        hedge = hyperedges[idx]
        hyperedges[idx] = frozenset({node_to_id[node] for node in hedge})
    
    hyperedges_to_keep = [
        idx for idx, hedge in enumerate(hyperedges) if len(hedge) > 1]
    hyperedges = [hyperedges[idx] for idx in hyperedges_to_keep]
    hyperedges_timestamps = [
        hyperedges_timestamps[idx] for idx in hyperedges_to_keep]
    
    return hyperedges, hyperedges_timestamps, len(node_to_id)

def split_hyperedges_according_to_time(hyperedges, hyperedges_timestamps):
    '''Split hyperedges according to timestamp. Sort hyperedges according to
    their timestamp and then split into three parts.

    Returns:
    List. A triplet consisting of hyperedges split into three parts.
    '''
    ground_idcs = [idx for idx, year in enumerate(hyperedges_timestamps) if year <= 2007]
    train_idcs = [idx for idx, year in enumerate(hyperedges_timestamps) if 2008 <= year <= 2013]
    test_idcs = [idx for idx, year in enumerate(hyperedges_timestamps) if 2013 <= year]
    ground_hyperedges = [hyperedges[idx] for idx in ground_idcs]
    train_hyperedges = [hyperedges[idx] for idx in train_idcs]
    test_hyperedges = [hyperedges[idx] for idx in test_idcs]
    return ground_hyperedges, train_hyperedges, test_hyperedges

def train_and_test_classifier(train_data, train_labels, test_data, test_labels):
    idcs = np.arange(len(train_data))
    np.random.shuffle(idcs)
    train_data = [train_data[idx] for idx in idcs]
    train_labels = [train_labels[idx] for idx in idcs]
    clf = LogisticRegression(penalty='l1')
    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(train_data)
    clf.fit(Xtrain, train_labels)
    Xtest = scaler.transform(test_data)
    Ptest = clf.predict_proba(Xtest)[:, 1]
    roc_score = roc_auc_score(test_labels, Ptest)
    return roc_score, clf

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
    args = parser.parse_args()

    if args.dataset == 'T2001-06':
        perpetrators, terror_plots = read_dataset()
        hyperedges, hyperedges_timestamps, total_nodes = (
            get_hyperedges_from_dataset(perpetrators, terror_plots))
        ground, train, test = split_hyperedges_according_to_time(
            hyperedges, hyperedges_timestamps)
    else:
        hyperedges, hyperedges_timestamps, total_nodes = read_chsim_dataset()
        ground, train, test = split_chsim_dataset(
            hyperedges, hyperedges_timestamps)

    Hg = utils.hyperedges_to_incidence_matrix(ground, total_nodes)
    Htrain = utils.hyperedges_to_incidence_matrix(train, total_nodes)
    Htest = utils.hyperedges_to_incidence_matrix(test, total_nodes)
    
    if args.classifier == 'FeatureBasedClassifier':
        print('Executing feature based classifier for hyperedge prediction')
        features = list(featurizer.FEATURE_MAP.keys())

        clfs = []
        roc_scores = []
        for i in range(args.nsamples):
            negative_hyperedges = sampler.sample_negative_hyperedges(
                ground, total_nodes, len(train))
            train_hyperedge_features, train_feature_matrix = (
                featurizer.hyperedge_featurizer(
                    train + negative_hyperedges, Hg, features=features))
            
            test_negative_hyperedges = sampler.sample_negative_hyperedges(
                ground + train, total_nodes, len(test))
            test_hyperedge_features, test_feature_matrix = (
                featurizer.hyperedge_featurizer(
                    test + test_negative_hyperedges,
                    np.concatenate([Hg.toarray(), Htrain.toarray()], axis=1),
                    features=features))
            
            train_labels = (
                [1 for _ in range(len(train))] +
                [0 for _ in range(len(negative_hyperedges))])
            test_labels = (
                [1 for _ in range(len(test))] +
                [0 for _ in range(len(test_negative_hyperedges))])
            
            roc_score, clf = train_and_test_classifier(
                train_feature_matrix, train_labels,
                test_feature_matrix, test_labels)
            clfs.append(clf)
            roc_scores.append(roc_score)
        coefs = np.array(
            [clf.coef_[0].tolist() for clf in clfs]).mean(axis=0).tolist()
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
