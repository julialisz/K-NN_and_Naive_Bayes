# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 2: k-NN and Naive Bayes
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: set of objects that are going to be compared (size: N1xD)
    :param X_train: set of objects compared against param X (size N2xD)
    Functions calculates Hamming distances between all objects from X and all object from X_train.
    Resulting distances are returned as matrix.
    :return: Matrix of distances between objects X and X_train (size: N1xN2)
    """
    '''x = X.toarray()
    x_train = X_train.toarray()
    N1 = len(x)
    N2 = len(x_train)
    dist_matrix = np.empty(shape=(N1, N2))
    for i in range(N1):
        for j in range(N2):
            dist_matrix[i][j] = (x[i] != x_train[j]).sum()
    return dist_matrix'''

    x = X.toarray().astype(int)
    x_train = X_train.toarray()
    x_train_trans = np.transpose(x_train).astype(int)
    return x.shape[1] - x @ x_train_trans - (1 - x) @ (1 - x_train_trans)


def sort_train_labels_knn(Dist, y):

    """
    Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
    Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
    :param Dist: Distance matrix between objects X and X_train (size: N1xN2)
    :param y: vector of labels (N2 elements)
    :return: Matrix of sorted class labels (use mergesort algorithm)
    """
    #order = Dist.argsort(kind='mergesort')
    #return y[order]
    output = []
    for row in Dist:
        zipped = zip(row, y)
        s = sorted(zipped, key=lambda x: x[0])
        temp = np.array([b for a, b in s])
        output.append(temp)
    output = np.array(output)

    return output


def p_y_x_knn(y, k):
    """
    Function calculates conditional probability p(y|x) for
    all classes and all objects from test set using KNN classifier
    :param y: matrix of sorted labels for training set (size: N1xN2)
    :param k: number of nearest neighbours
    :return: matrix of probabilities for objects X
    """
    topics = set([i for i in range(0, 4)])
    #for i in y:
        #for j in i:
            #topics.add(j)

    prob_matrix = np.zeros(shape=(len(y), len(topics)))
    for t in topics:
        for i in range(len(y)):
            total = 0
            for j in range(k):
                if(y[i][j] == t):
                    total+=1
                #total += 1 if y[i][j] == t else 0
            prob_matrix[i][t] = total/k

    return prob_matrix


def classification_error(p_y_x, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels (size: 1xN).
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """
    N1 = len(p_y_x)
    m = len(p_y_x[0])
    result = 0
    for i in range(N1):
        if (m - np.argmax(p_y_x[i][::-1]) - 1) != y_true[i]:
            result += 1
    return result / N1


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: validation data (size: N1xD)
    :param Xtrain: training data (size: N2xD)
    :param yval: class labels for validation data (size: 1xN1)
    :param ytrain: class labels for training data (size: 1xN2)
    :param k_values: values of parameter k that must be evaluated
    :return: function performs model selection with knn and returns tuple (best_error,best_k,errors), where best_error is the lowest
    error, best_k - value of k parameter that corresponds to the lowest error, errors - list of error values for
    subsequent values of k (elements of k_values)
    """
    hammDist = hamming_distance(Xval, Xtrain)
    sortedTrainLabels = sort_train_labels_knn(hammDist, ytrain)
    errors = []
    for k in range(len(k_values)):
        error = classification_error(p_y_x_knn(sortedTrainLabels, k_values[k]), yval)
        errors.append(error)
    best_k = k_values[np.argmin(errors)]
    minError = min(errors)
    return minError, best_k, np.array(errors)


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: labels for training data (size: 1xN)
    :return: function calculates distribution a priori p(y) and returns p_y - vector of a priori probabilities (size: 1xM)
    """
    pi = np.bincount(ytrain)/len(ytrain)
    return pi


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: training data (size: NxD)
    :param ytrain: class labels for training data (size: 1xN)
    :param a: parameter a of Beta distribution
    :param b: parameter b of Beta distribution
    :return: Function calculates probality p(x|y), assuming that x is binary variable and elements of x 
    are independent from each other. Function returns matrix p_x_y (size: MxD).
    """
    xtrain = Xtrain.toarray()
    y_set = set()
    for y in ytrain:
        y_set.add(y)
    theta = []
    for k in y_set:
        temp = []
        for d in range(xtrain.shape[1]):
            nominator = 0
            for n in range(len(ytrain)):
                nominator += 1 if (ytrain[n] == k and xtrain[n][d] == 1) else 0

            denominator = 0
            for n in range(len(ytrain)):
                denominator += 1 if ytrain[n] == k else 0

            temp.append((nominator + a - 1) / (denominator + a + b - 2))
        theta.append(np.array(temp))
    theta = np.array(theta).reshape(len(y_set), xtrain.shape[1])
    return theta


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: vector of a priori probabilities (size: 1xM)
    :param p_x_1_y: probability distribution p(x=1|y) (matrix, size: MxD)
    :param X: data for probability estimation, matrix (matrix, size: NxD)
    :return: function calculates probability distribution p(y|x) for each class with the use of Naive Bayes classifier.
    Function returns matrix p_y_x (size: NxM).
    """
    x = X.toarray()
    N = x.shape[0]
    D = x.shape[1]
    M = p_x_1_y.shape[0]
    output = []
    for n in range(N):
        temp = []
        e = 0
        for k in range(M):
            product = 1
            for d in range(D):
                theta = p_x_1_y[k][d]
                x_d = int(x[n][d])
                product *= (theta ** x_d) * ((1 - theta) ** (1 - x_d))
            e += product * p_y[k]
            temp.append(product)
        temp = np.array(temp)
        for k in range(M):
            output.append(temp[k] * p_y[k] / e)
    output = np.array(output).reshape((N, M))
    return output


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: training set (size: N2xD)
    :param Xval: validation set (size: N1xD)
    :param ytrain: class labels for training data (size: 1xN2)
    :param yval: class labels for validation data (size: 1xN1)
    :param a_values: list of parameters a (Beta distribution)
    :param b_values: list of parameters b (Beta distribution)
    :return: Function performs a model selection for Naive Bayes. It selects the best values of a i b parameters.
    Function returns tuple (error_best, best_a, best_b, errors) where best_error is the lowest error,
    best_a - a parameter that corresponds to the lowest error, best_b - b parameter that corresponds to the lowest error,
    errors - matrix of errors all pairs (a,b)
    """
    errors = np.ones((len(a_values), len(b_values)))
    pi = estimate_a_priori_nb(ytrain)
    best_a = 0
    best_b = 0
    best_error = np.inf
    for i in range(len(a_values)):
        for j in range(len(b_values)):
            theta = estimate_p_x_y_nb(Xtrain, ytrain, a_values[i], b_values[j])
            error = classification_error(p_y_x_nb(pi, theta, Xval), yval)
            errors[i][j] = error
            if error < best_error:
                best_a = a_values[i]
                best_b = b_values[j]
                best_error = error
    return best_error, best_a, best_b, errors
