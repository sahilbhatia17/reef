import numpy as np
import scipy
import json
import pickle as pkl

from scipy import sparse

def parse_user_data(userfilename):
    # load the file
    with open(userfilename, "rb") as ufl:
        user_info = pkl.load(ufl)
        # kinda hacky, but hopefully will work for now
        if isinstance(user_info[-1], str):
            user_info = user_info[0]
    # operate with the tuple sequence collected in the user study
    user_indices = []
    for idx, plot, label in user_info:
        user_indices.append(idx)
    return np.array(idx)

def parse_file(filename):

    def parse(filename):
        movies = []
        with open(filename) as f:
            for line in f:
                obj = json.loads(line)
                movies.append(obj)
        return movies

    f = parse(filename)
    gt = []
    plots = []
    idx = []
    for i,movie in enumerate(f):
        genre = movie['Genre']
        if 'Action' in genre and 'Romance' in genre:
            continue
        elif 'Action' in genre:
            if movie['Plot'] != 'N/A':
                plots = plots+[movie['Plot']]
                gt.append(1)
                idx.append(i)
        elif 'Romance' in genre:
            if movie['Plot'] != 'N/A':
                plots = plots+[movie['Plot']]
                gt.append(-1)
                idx.append(i)
        else:
            continue  
    
    return np.array(plots), np.array(gt)

def split_data(X, plots, y):
    import sklearn.model_selection
    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    num_test = 500

    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    # split dev/test
    test_ratio = 0.2
    X_tr, X_te, y_tr, y_te, plots_tr, plots_te = \
        sklearn.model_selection.train_test_split(X_train, y_train, plots_train, test_size = test_ratio)

    return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()), \
        np.array(y_tr), np.array(y_te), np.array(y_test), plots_tr, plots_te, plots_test

def split_user_data(X, plots, y, user_idxs):
    import sklearn.model_selection
    
    # get the validation set first, from the user indices
    val_train = []
    val_plots = []
    val_labels = []
    for idx in user_idxs:
        val_train.append(X.pop(idx))
        val_plots.append(plots.pop(idx))
        val_labels.append(y.pop(idx))
    
    # now, construct test and train via same arbitrary split
    num_test = 500
    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    return np.array(X_train), np.array(val_train), np.array(X_test), \
    np.array(y_train), np.array(val_labels), np.array(y_test), plots_train, val_plots, plots_test


class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='./data/imdb/'):
        from sklearn.feature_extraction.text import CountVectorizer

        #Parse Files
        plots, labels = parse_file(data_path+'budgetandactors.txt')
        #read_plots('imdb_plots.tsv')

        #Featurize Plots  
        vectorizer = CountVectorizer(min_df=1, binary=True, \
            decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        X = vectorizer.fit_transform(plots)
        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]

        #Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = split_data(X, plots, labels)

        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
            np.array(train_ground), np.array(val_ground), np.array(test_ground), \
            train_plots, val_plots, test_plots
    
    def load_data_from_user(self, dataset, user_path, data_path='./data/imdb/'):
        from sklearn.feature_extraction.text import CountVectorizer

        #Parse Files
        plots, labels = parse_file(data_path+'budgetandactors.txt')
        #read_plots('imdb_plots.tsv')

        #Featurize Plots  
        vectorizer = CountVectorizer(min_df=1, binary=True, \
            decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        X = vectorizer.fit_transform(plots)
        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]

        labeled_indices = parse_user_data(user_path)

        #Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = split_user_data(X, plots, labels, labeled_indices)

        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
            np.array(train_ground), np.array(val_ground), np.array(test_ground), \
            train_plots, val_plots, test_plots
        
