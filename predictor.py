from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.sparse as sp

import negative_sampler as sampler
import hyperedge_features as featurizer
import spectral_hyper_clustering as spectral
import utils

def train_and_test_classifier(
        train_data, train_labels, test_data, test_labels, use_scaler=True):
    idcs = np.arange(len(train_data))
    np.random.shuffle(idcs)
    train_data = [train_data[idx] for idx in idcs]
    train_labels = [train_labels[idx] for idx in idcs]
    clf = LogisticRegression(penalty='l1')

    if use_scaler:
        scaler = MinMaxScaler()
        Xtrain = scaler.fit_transform(train_data)
    else:
        Xtrain = train_data
    clf.fit(Xtrain, train_labels)

    if use_scaler:
        Xtest = scaler.transform(test_data)
    else:
        Xtest = test_data

    Ptest = clf.predict_proba(Xtest)[:, 1]
    roc_score = roc_auc_score(test_labels, Ptest)
    return roc_score, clf

def train_feature_based_classifier(
        ground, train, test, features, total_nodes, node_active=None):
    Hg = utils.hyperedges_to_incidence_matrix(ground, total_nodes)
    Hcomb = utils.hyperedges_to_incidence_matrix(ground + train, total_nodes)
    
    if node_active is None:
        print('Sampling Train Negative Hyperedges')
        negative_hyperedges = sampler.sample_negative_hyperedges(
            ground, total_nodes, len(train))
        print('Calculating Training data features')
        train_hyperedge_features, train_feature_matrix = (
            featurizer.hyperedge_featurizer(
                train + negative_hyperedges, Hg, features=features))
        
        print('Samping Test Negative Hyperedges')
        test_negative_hyperedges = sampler.sample_negative_hyperedges(
            ground + train, total_nodes, len(test))
        print('Calculating Test data features')
        test_hyperedge_features, test_feature_matrix = (
            featurizer.hyperedge_featurizer(
                test + test_negative_hyperedges, Hcomb, features=features))
        
        train_labels = (
            [1 for _ in range(len(train))] +
            [0 for _ in range(len(negative_hyperedges))])
        test_labels = (
            [1 for _ in range(len(test))] +
            [0 for _ in range(len(test_negative_hyperedges))])
        
    else:
        print('Calculating Training data features')
        train_hyperedge_features, train_feature_matrix = (
            featurizer.hyperedge_featurizer(train, Hg, features=features))
        
        print('Calculating Test data features')
        test_hyperedge_features, test_feature_matrix = (
            featurizer.hyperedge_featurizer(test, Hcomb, features=features))
        print(train_feature_matrix.shape)
        train_labels = [
            1 if all(node_active[node] for node in hedge) else 0
            for hedge in train]
        test_labels = [
            1 if all(node_active[node] for node in hedge) else 0
            for hedge in test]
    
    roc_score, clf = train_and_test_classifier(
        train_feature_matrix, train_labels,
        test_feature_matrix, test_labels)
    return roc_score, clf

def train_spectral_classifier(
    ground, train, test, total_nodes, dim, emb_type, pool='max', node_active=None):
    
    if node_active is None:
        print('Sampling Train Negative Hyperedges')
        negative_hyperedges = sampler.sample_negative_hyperedges(
            ground, total_nodes, len(train))
        H = utils.hyperedges_to_incidence_matrix(
            ground + train + negative_hyperedges, total_nodes)
        print('Calculating Training data features')
        if emb_type == 'dual':
            train_features = spectral.get_hyperedge_embeddings_from_dual(
                H, dim=dim)
            base_val = len(ground)
            train_features = train_features[base_val: base_val + len(train), :]
        elif emb_type == 'primal':
            train_features = spectral.get_hyperedge_embeddings_from_nodes(
                H, train, dim=dim, pool=pool)

        
        print('Samping Test Negative Hyperedges')
        test_negative_hyperedges = sampler.sample_negative_hyperedges(
            ground + train, total_nodes, len(test))
        H = utils.hyperedges_to_incidence_matrix(
            ground + train + test + test_negative_hyperedges, total_nodes)
        
        print('Calculating Test data features')
        if emb_type == 'dual':
            test_features = spectral.get_hyperedge_embeddings_from_dual(
                H, dim=dim)
            base_val = len(ground) + len(train)
            test_features = test_features[base_val: base_val + len(test), :]
        elif emb_type == 'primal':
            test_features = spectral.get_hyperedge_embeddings_from_nodes(
                H, test, dim=dim, pool=pool)
        
        train_labels = (
            [1 for _ in range(len(train))] +
            [0 for _ in range(len(negative_hyperedges))])
        test_labels = (
            [1 for _ in range(len(test))] +
            [0 for _ in range(len(test_negative_hyperedges))])
        
    else:
        print('Calculating Training data features')
        H = utils.hyperedges_to_incidence_matrix(ground + train, total_nodes)
        if emb_type == 'dual':
            train_features = spectral.get_hyperedge_embeddings_from_dual(
                H, dim=dim)
            train_features = train_features[-len(train):, :]
        elif emb_type == 'primal':
            train_features = spectral.get_hyperedge_embeddings_from_nodes(
                H, train, dim=dim, pool=pool)
        
        print('Calculating Test data features')
        H = utils.hyperedges_to_incidence_matrix(
            ground + train + test, total_nodes)
        if emb_type == 'dual':
            test_features = spectral.get_hyperedge_embeddings_from_dual(
                H, dim=dim)
            test_features = test_features[-len(test):, :]
        elif emb_type == 'primal':
            test_features = spectral.get_hyperedge_embeddings_from_nodes(
                H, test, dim=dim, pool=pool)
        
        train_labels = [
            1 if all(node_active[node] for node in hedge) else 0
            for hedge in train]
        test_labels = [
            1 if all(node_active[node] for node in hedge) else 0
            for hedge in test]
    
    roc_score, clf = train_and_test_classifier(
        train_features, train_labels, test_features, test_labels,
        use_scaler=False)
    return roc_score, clf
