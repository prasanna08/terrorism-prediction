import utils
import hyperedge_features as featurizer
import negative_sampler as sampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocess import MinMaxScaler
from sklearn.metrics import roc_auc_score

def read_dataset():
    '''Read dataset from file.
    
    Returns:
    DataFrame. Pandas dataframe containing data from file.
    '''
    pass

def get_hyperedges_from_dataset(data):
    '''Create a list of hyperedges from the data. Each hyperedge is a frozenset
    of the nodes it contains.

    Returns:
    hyperedges: List. List of all hyperedges
    hyperedges_timestamps: List. Timestamp associated with each hyperedge.
        Timestamps should be in same order as that of hyperedges.
    total_nodes: Int. Total number of nodes in hypergraph.
    '''
    pass

def split_hyperedges_according_to_time(hyperedges, hyperedges_timestamps):
    '''Split hyperedges according to timestamp. Sort hyperedges according to
    their timestamp and then split into three parts.

    Returns:
    List. A triplet consisting of hyperedges split into three parts.
    '''
    pass

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
    data = read_dataset()
    hyperedges, hyperedges_timestamps, total_nodes = (
        get_hyperedges_from_dataset(data))
    ground, train, test = split_hyperedges_according_to_time(
        hyperedges, hyperedges_timestamps)
    Hg = utils.hyperedges_to_incidence_matrix(ground, total_nodes)
    Htrain = utils.hyperedges_to_incidence_matrix(train, total_nodes)
    Htest = utils.hyperedges_to_incidence_matrix(test, total_nodes)

    if args.classifier == 'FeatureBasedClassifier':
        print('Executing feature based classifier for hyperedge prediction')
        negative_hyperedges = sampler.sample_negative_hyperedges(
            ground, total_nodes, len(train))
        train_hyperedge_features, train_feature_matrix = (
            featurizer.hyperedge_featurizer(
                train + negative_hyperedges, Hg, features=None))
        test_negative_hyperedges = sampler.sample_negative_hyperedges(
            ground + train, total_nodes, len(test))
        test_hyperedge_features, test_feature_matrix = (
            featurizer.hyperedge_featurizer(
                test + test_negative_hyperedges,
                np.concatenate([Hg Htrain], axis=1), features=None))
        train_labels = [1 for _ in range(len(train))] + [0 for _ in range(len(negative_hyperedges))]
        test_labels = [1 for _ in range(len(test))] + [0 for _ in range(len(test_negative_hyperedges))]
        roc_score, clf = train_and_test_classifier(
            train_feature_matrix, train_labels,
            test_feature_matrix, test_labels)
