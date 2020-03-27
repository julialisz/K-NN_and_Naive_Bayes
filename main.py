# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 2: k-NN and Naive Bayes
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------------- DO NOT MODIFY THIS FILE --------------------------
# --------------------------------------------------------------------------

import pickle
import warnings
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from content import (classification_error, estimate_a_priori_nb, estimate_p_x_y_nb,
                     hamming_distance, model_selection_knn, model_selection_nb, p_y_x_knn, p_y_x_nb,
                     sort_train_labels_knn)
from test import TestRunner


def plot_a_b_errors(errors, a_points, b_points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(errors)
    fig.colorbar(cax)
    plt.title("Model selection for NB")
    ax.set_xticklabels([''] + a_points)
    ax.set_yticklabels([''] + b_points)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_tick_params(labelbottom=True, labeltop=False, top=False)
    ax.set_xlabel('Parameter b')
    ax.set_ylabel('Parameter a')
    plt.draw()
    plt.waitforbuttonpress(0)


def plot_error_NB_KNN(error_NB, error_KNN):
    plt.figure()
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'
    plt.style.use(['dark_background'])
    labels = ["Naive Bayes", "KNN"]
    data = [error_NB, error_KNN]

    xlocations = np.array(range(len(data))) + 0.5
    width = 0.5
    plt.bar(xlocations, data, width=width, color='#FFCC55')
    plt.xticks(xlocations, labels)
    plt.xlim(0, xlocations[-1] + width * 2 - .5)
    plt.title("Model comparison - classification error")
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.draw()
    plt.waitforbuttonpress(0)


def classification_KNN_vs_no_neighbours(xs, ys):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'
    plt.style.use(['dark_background'])
    plt.xlabel('Number of neighbours k')
    plt.ylabel('Classification error')
    plt.title("Model selection for k-NN")

    plt.plot(xs, ys, 'r-', color='#FFCC55')
    plt.draw()
    plt.waitforbuttonpress(0)


def word_cloud(frequencies, title):
    from wordcloud import WordCloud
    wordcloud = WordCloud(font_path='assets/DroidSansMono.ttf',
                          relative_scaling=1.0).generate_from_frequencies(frequencies)
    plt.title(title)
    plt.imshow(wordcloud)
    plt.axis("off")
    return wordcloud


def word_clouds(list_of_frequencies, topics):
    fig = plt.figure(num='Distribution of words for each document class and NB model')
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'
    plt.style.use(['dark_background'])
    for idx, (topic, frequencies) in enumerate(zip(topics, list_of_frequencies)):
        location = 221 + idx
        plt.subplot(location)
        wordcloud = word_cloud(frequencies, topic)
        plt.axis("off")
        plt.imshow(wordcloud)
    plt.draw()
    plt.waitforbuttonpress(0)


def run_unittests():
    test_runner = TestRunner()
    results = test_runner.run()
    if results.failures or results.errors:
        exit()
    sleep(0.1)


def load_data():
    PICKLE_FILE_PATH = 'data.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pickle.load(f)


def run_training():
    data = load_data()

    # KNN model selection
    k_values = range(1, 201, 2)
    print('\n------------- Model selection for KNN -------------')
    print('-------------------- Values k: 1, 3, ..., 200 -----------------------')
    print('--------------------- Calculation may take up to 1 min ------------------------')

    error_best, best_k, errors = model_selection_knn(data['Xval'],
                                                     data['Xtrain'],
                                                     data['yval'],
                                                     data['ytrain'],
                                                     k_values)
    print('The best k: {num1} and the best error: {num2:.4f}'.format(num1=best_k, num2=error_best))
    print('\n--- Press any key to continue ---')
    classification_KNN_vs_no_neighbours(k_values, errors)
    a_values = [1, 3, 10, 30, 100, 300, 1000]
    b_values = [1, 3, 10, 30, 100, 300, 1000]

    print('\n----------------- Model selection for a and b --------------------')
    print('--------- Values a and b: 1, 3, 10, 30, 100, 300, 1000 -----------------')
    print('--------------------- Calculation may take up to 1 min ------------------------')

    # NB model selection
    error_best, best_a, best_b, errors = model_selection_nb(data['Xtrain'], data['Xval'],
                                                            data['ytrain'],
                                                            data['yval'], a_values, b_values)

    print('The best a: {}, b: {} and the best error: {:.4f}'.format(best_a, best_b, error_best))
    print('\n--- Press any key to continue ---')
    plot_a_b_errors(errors, a_values, b_values)
    p_x_y = estimate_p_x_y_nb(data['Xtrain'], data['ytrain'], best_a, best_b)

    classes_no = p_x_y.shape[0]
    print('\n------ Visualization of most popular words for each class ------')
    print('-- These are words that are most probable for each class and NB model --')

    try:
        groupnames = data['groupnames']
        words = {}
        for x in range(classes_no):
            indices = np.argsort(p_x_y[x, :])[::-1][:50]
            words[groupnames[x]] = {word: prob for word, prob in
                                    zip(data['wordlist'][indices], p_x_y[x, indices])}
        word_clouds(words.values(), words.keys())
    except Exception:
        print('--- A problem with wordcloud library --- ')

    print('\n--- Press any key to continue ---')

    print('\n---------------- Comparison of KNN and NB errors ---------------------')

    Dist = hamming_distance(data['Xtest'], data['Xtrain'])
    y_sorted = sort_train_labels_knn(Dist, data['ytrain'])
    p_y_x = p_y_x_knn(y_sorted, best_k)
    error_KNN = classification_error(p_y_x, data['ytest'])

    p_y = estimate_a_priori_nb(data['ytrain'])
    p_y_x = p_y_x_nb(p_y, p_x_y, data['Xtest'])
    error_NB = classification_error(p_y_x, data['ytest'])

    plot_error_NB_KNN(error_NB, error_KNN)
    print('\n--- Press any key to continue ---')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_unittests()
    run_training()
