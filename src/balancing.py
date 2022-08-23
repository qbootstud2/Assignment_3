import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

from src.utils import shuffle_2d
from src.models import scores_logistic_regression, scores_random_forest, scores_neural_network


def score_model_unbalanced(model, X, Y, name, n_splits=5, verbose=True):
    """ This function creates folds manually and trains the model on the unbalanced dataset.
    The output is an array with the model predictions."""

    # Define cross-validation kfolds
    kfold = KFold(n_splits=n_splits, shuffle=False)

    # Define list to save prediction scores
    scores = []

    # Enumerate the splits and summarize the distributions
    for train_index, test_index in kfold.split(X, Y):

        # Divide data in train and test sets
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = Y[train_index], Y[test_index]

        # Count number of events in the train and test sets
        n_sg_train, n_bg_train = len(train_X[train_Y == 1]), len(train_X[train_Y == 0])
        n_sg_test, n_bg_test = len(test_Y[test_Y == 1]), len(test_Y[test_Y == 0])

        # Shuffle input arrays
        train_X, train_Y = shuffle_2d(train_X, train_Y)

        if verbose:
            print('Training: sg = %d, bg = %d' % (n_sg_train, n_bg_train))
            print('Testing:  sg = %d, bg = %d\n' % (n_sg_test, n_bg_test))

        # Fit the model on the undersampled train data and get predictions on test data
        if name == "Regression":
            score = scores_logistic_regression(model, train_X, train_Y, test_X)
        elif name == "Random Forest":
            score = scores_random_forest(model, train_X, train_Y, test_X)
        elif name == "NN":
            score = scores_neural_network(model, train_X, train_Y, test_X,
                                          epochs=300, batch_size=128, verbose=0, visualize_training=True)
        else:
            score = []
            print('Error: no model has been specified')

        # Save scores on the test set
        scores.append(score)

    scores = np.concatenate(([el for el in scores]), axis=0)

    return scores


def score_model_undersampling(model, X, Y, name, n_splits=5, verbose=True):
    """ This function creates folds manually and does a random undersampling within each fold.
    The output is an array with the model predictions."""

    # Define cross-validation kfolds
    kfold = KFold(n_splits=n_splits, shuffle=False)

    # Define list to save prediction scores
    scores = []

    # Enumerate the splits and summarize the distributions
    for train_index, test_index in kfold.split(X, Y):

        # Divide data in train and test sets
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = Y[train_index], Y[test_index]

        # Undersample the train set
        train_sg = train_X[train_Y == 1]
        train_bg = np.random.permutation(train_X[train_Y == 0])[:len(train_sg)]

        # Count number of events in the train and test sets
        n_sg_train, n_bg_train = len(train_sg), len(train_bg)
        n_sg_test, n_bg_test = len(test_Y[test_Y == 1]), len(test_Y[test_Y == 0])

        # Build undersampled train and test sets
        train_X = np.concatenate((train_sg, train_bg), axis=0)
        train_Y = np.concatenate((np.ones(n_sg_train), np.zeros(n_bg_train)), axis=0)

        # Shuffle input arrays
        train_X, train_Y = shuffle_2d(train_X, train_Y)

        if verbose:
            print('Training: sg = %d, bg = %d' % (n_sg_train, n_bg_train))
            print('Testing:  sg = %d, bg = %d\n' % (n_sg_test, n_bg_test))

        # Fit the model on the undersampled train data and get predictions on test data
        if name == "Regression":
            score = scores_logistic_regression(model, train_X, train_Y, test_X)
        elif name == "Random Forest":
            score = scores_random_forest(model, train_X, train_Y, test_X)
        elif name == "NN":
            score = scores_neural_network(model, train_X, train_Y, test_X,
                                          epochs=300, batch_size=128, verbose=0, visualize_training=True)
        else:
            score = []
            print('Error: no model has been specified')

        # Save scores on the test set
        scores.append(score)

    scores = np.concatenate(([el for el in scores]), axis=0)

    return scores


def score_model_oversampling(model, X, Y, name, n_splits=5, verbose=True):
    """ This function creates folds manually and does a random oversampling within each fold.
    The output is an array with the model predictions."""

    # Define cross-validation kfolds
    kfold = KFold(n_splits=n_splits, shuffle=False)

    # Define SMOTE to make the oversampling
    smoter = SMOTE(random_state=42)

    # Define list to save prediction scores
    scores = []

    # Enumerate the splits and summarize the distributions
    for train_index, test_index in kfold.split(X, Y):

        # Divide data in training and testing sets
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = Y[train_index], Y[test_index]

        # Oversample train set
        train_X, train_Y = smoter.fit_resample(train_X, train_Y)

        # Count number of events in the train and test sets
        n_sg_train, n_bg_train = len(train_X[train_Y == 1]), len(train_X[train_Y == 0])
        n_sg_test, n_bg_test = len(test_Y[test_Y == 1]), len(test_Y[test_Y == 0])

        # Shuffle input arrays
        train_X, train_Y = shuffle_2d(train_X, train_Y)

        if verbose:
            print('Training: sg = %d, bg = %d' % (n_sg_train, n_bg_train))
            print('Testing:  sg = %d, bg = %d\n' % (n_sg_test, n_bg_test))

        # Fit the model on the oversampled train data and get predictions on test data
        if name == "Regression":
            score = scores_logistic_regression(model, train_X, train_Y, test_X)
        elif name == "NN":
            score = scores_neural_network(model, train_X, train_Y, test_X,
                                          epochs=50, batch_size=128, verbose=0, visualize_training=True)
        else:
            score = []
            print('Error: no model has been specified')

        # Save scores on the test set
        scores.append(score)

    scores = np.concatenate(([el for el in scores]), axis=0)

    return scores
