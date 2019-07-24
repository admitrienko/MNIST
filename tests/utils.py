import operator
import plotly.plotly as py
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Flatten, Embedding, Dense, Activation, Input
import matplotlib.pyplot as plt
import gzip
import random
from keras import backend as K
from sklearn import manifold
from sklearn.svm import LinearSVC
import math
from itertools import combinations
from itertools import permutations
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# sample from isotropic Gaussian data
def sample_random_data(dimensions):

    random_data = np.ones((1000, 100, 8))

    for i in range(1000):
        for j in range(100):
            for k in range(8):
                random_data[i, j, k] = np.random.normal()

    return random_data


def get_accuracy(predictions):
    mag_predictions = predictions[1].T
    parity_predictions = predictions[0].T

    mag_pred = np.ones(len(mag_predictions.T), dtype=object)
    par_pred = np.ones(len(mag_predictions.T), dtype=object)

    for mag in range(len(mag_predictions.T)):

        if mag_predictions[0][mag] < 0.5:
            mag_pred[mag] = "low"
        else:
            mag_pred[mag] = "high"

    for par in range(len(parity_predictions.T)):

        if parity_predictions[0][par] < 0.5:
            par_pred[par] = "even"
        else:
            par_pred[par] = "odd"

    mag_count = 0
    parity_count = 0
    for pred in range(len(mag_predictions.T)):

        if mag_pred[pred] == test_magnitudes[pred]:
            mag_count += 1

        if par_pred[pred] == test_parities[pred]:
            parity_count += 1

    mag_accuracy = mag_count / len(mag_predictions.T)
    parity_accuracy = parity_count / len(mag_predictions.T)

    return (mag_accuracy, parity_accuracy)


def cov_3D(data):

    dim_1 = data.shape[0]
    dim_2 = data.shape[1]
    dim_3 = data.shape[2]

    data_1 = data

    data_2 = np.swapaxes(data, 0, 1)

    data_3 = np.swapaxes(data, 0, 2)

    # shape (1, (2*3))
    sigma_1_data = np.reshape(data_1, (dim_1, (dim_2 * dim_3)), order="C")

    # shape (2, (1*3))
    sigma_2_data = np.reshape(data_2, (dim_2, (dim_1 * dim_3)), order="C")

    # shape (3, (1*2))
    sigma_3_data = np.reshape(data_3, (dim_3, (dim_2 * dim_1)), order="C")

    sigma_1 = np.zeros((dim_1, dim_1))

    for a in range(dim_1):

        for b in range(dim_1):

            sigma_1[a, b] = np.matmul(sigma_1_data[a, :], sigma_1_data[b, :].T)

    sigma_2 = np.zeros((dim_2, dim_2))

    for a in range(dim_2):

        for b in range(dim_2):

            sigma_2[a, b] = np.matmul(sigma_2_data[a, :], sigma_2_data[b, :].T)

    sigma_3 = np.zeros((dim_3, dim_3))

    for a in range(dim_3):

        for b in range(dim_3):

            sigma_3[a, b] = np.matmul(sigma_3_data[a, :], sigma_3_data[b, :].T)

    sigma_3_trace = np.trace(sigma_3)
    sigma_2_trace = np.trace(sigma_2)
    sigma_1_trace = np.trace(sigma_1)

    # print(sigma_1_trace)
    # print(sigma_2_trace)
    # print(sigma_3_trace)

    return sigma_1, sigma_2, sigma_3


def get_dichotomies():
    dichotomies = []
    digits = [1, 2, 3, 4, 5, 6, 7, 8]
    first_digit = 1

    comb = combinations(digits, 4)

    for d in comb:
        d_list = list(d)
        d_list2 = []
        for num in digits:
            if num not in d_list:
                d_list2.append(num)

        dichotomies.append((d_list, d_list2))

    dichotomies = dichotomies[0:35]

    return dichotomies


def shuffle_dataset(current_data, current_labels):

    T = current_data.shape[0]
    N = current_data.shape[1]
    C = current_data.shape[2]

    index_locations = np.ones(N)

    new_dataset = np.ones((T * C, N))
    new_labels = np.ones(T * C)

    counter = 0
    for i in range(T):
        start = i

        for j in range(C):

            new_dataset[counter] = current_data[start + j * T]
            new_labels[counter] = current_labels[start + j * T]

            index_locations

            counter += 1

    return new_dataset, new_labels, index_locations


def plot_MDS(input_dataset, input_labels, number_points=800, title="MDS Plot"):

    dataset, labels = shuffle_dataset(input_dataset, input_labels)

    colors = np.ones(len(labels), dtype=object)
    index = 0
    for label in labels:
        lab = int(label)
        if lab == 1 or lab == 3:
            color = "firebrick"
        if lab == 2 or lab == 4:
            color = "lightskyblue"
        if lab == 5 or lab == 7:
            color = "lightcoral"
        if lab == 6 or lab == 8:
            color = "darkblue"

        colors[index] = color
        index += 1

    embedding = manifold.MDS(n_components=2)

    train_image_transformed = embedding.fit_transform(layer1_output[1200:1600])
    plt.scatter(
        train_image_transformed[:, 0],
        train_image_transformed[:, 1],
        c=colors[1200:1600],
    )
    plt.title(title, fontsize=30)
    red_patch = mpatches.Patch(color="firebrick", label="1,3")
    pink_patch = mpatches.Patch(color="lightcoral", label="5,7")
    blue_patch = mpatches.Patch(color="lightskyblue", label="2,4")
    navy_patch = mpatches.Patch(color="darkblue", label="6,8")
    plt.legend(
        handles=[red_patch, pink_patch, blue_patch, navy_patch],
        loc=(1, 0),
        prop={"size": 20},
    )

    plt.xticks([])

    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xlabel("MDS 1", fontsize=24)
    plt.ylabel("MDS 2", fontsize=24)

    plt.show()


def plot_PCA(dataset, labels, title="PCA Plot"):

    pca = PCA(n_components=3)

    pca_data = pca.fit_transform(data)

    colors = np.ones(len(labels), dtype=object)
    index = 0
    for label in labels:
        lab = int(label)
        if lab == 1 or lab == 3:
            color = "firebrick"
        if lab == 2 or lab == 4:
            color = "lightskyblue"
        if lab == 5 or lab == 7:
            color = "lightcoral"
        if lab == 6 or lab == 8:
            color = "darkblue"

        colors[index] = color
        index += 1

    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors)
    plt.title(title)

    red_patch = mpatches.Patch(color="firebrick", label="1,3")
    pink_patch = mpatches.Patch(color="lightcoral", label="5,7")
    blue_patch = mpatches.Patch(color="lightskyblue", label="2,4")
    navy_patch = mpatches.Patch(color="darkblue", label="6,8")
    plt.legend(handles=[red_patch, pink_patch, blue_patch, navy_patch])

    plt.xticks([])
    plt.yticks([])
    plt.show()


def cosine(v1, v2):

    dot_product = sum((a * b) for a, b in zip(v1, v2))

    length_v1 = math.sqrt(sum((a * b) for a, b in zip(v1, v1)))

    length_v2 = math.sqrt(sum((a * b) for a, b in zip(v2, v2)))

    return dot_product / (length_v1 * length_v2)


def ccgp(data, dichotomy=([1, 2, 3, 4], [5, 6, 7, 8]), labels=[]):
    labels1 = dichotomy
    label_1a = dichotomy[0]
    label_1b = dichotomy[1]

    accuracy = []

    label_1a_combs = combinations(label_1a, 3)
    label_1b_combs = combinations(label_1b, 3)

    label_1a_set = []
    label_1b_set = []

    for a in label_1a_combs:

        label_1a_set.append(a)

    for b in label_1b_combs:

        label_1b_set.append(b)

    for train_a in label_1a_set:

        for train_b in label_1b_set:

            label_2a = labels1[0]  # dichotomy a
            label_2b = labels1[1]  # dichotomy b

            target1 = []
            target2 = []

            data_train = []

            data_test = []

            for lab in range(len(labels)):
                if int(labels[lab]) in label_2a:  # [1,2,3,4]

                    if int(labels[lab]) in train_a:  # [1,2,3]

                        target1.append(0)

                        data_train.append(data[lab])

                    else:
                        target2.append(0)  # [4]

                        data_test.append(data[lab])

                else:  # [5,6,7,8]

                    if int(labels[lab]) in train_b:

                        target1.append(1)
                        data_train.append(data[lab])

                    else:
                        target2.append(1)

                        data_test.append(data[lab])

            target1 = np.array(target1)
            target2 = np.array(target2)

            data_train = np.array(data_train)
            data_test = np.array(data_test)

            # fit on training data
            svc1 = LinearSVC(C=1.0, max_iter=50000)

            svc1.fit(data_train, target1.T)

            # test on testing data
            acc = svc1.score(data_test, target2.T)
            accuracy.append(acc)

    return np.mean(accuracy)


def p_score(data, dichotomy=([1, 2, 3, 4], [5, 6, 7, 8]), labels=[]):

    plane_cosine = []

    label1b = dichotomy[1][0]
    label2b = dichotomy[1][1]
    label3b = dichotomy[1][2]
    label4b = dichotomy[1][3]

    for perm in permutations(dichotomy[0]):

        label1a = perm[0]
        target1 = []
        data1 = []

        label2a = perm[1]
        target2 = []
        data2 = []

        label3a = perm[2]
        target3 = []
        data3 = []

        label4a = perm[3]
        target4 = []
        data4 = []

        for lab in range(len(labels)):
            if int(labels[lab]) == label1a:

                target1.append(0)

                data1.append(data[lab])

            if int(labels[lab]) == label1b:

                target1.append(1)

                data1.append(data[lab])

            if int(labels[lab]) == label2a:

                target2.append(0)

                data2.append(data[lab])

            if int(labels[lab]) == label2b:

                target2.append(1)

                data2.append(data[lab])

            if int(labels[lab]) == label3a:

                target3.append(0)

                data3.append(data[lab])

            if int(labels[lab]) == label3b:

                target3.append(1)

                data3.append(data[lab])

            if int(labels[lab]) == label4a:

                target4.append(0)

                data4.append(data[lab])

            if int(labels[lab]) == label4b:

                target4.append(1)

                data4.append(data[lab])

        target1 = np.array(target1)
        target2 = np.array(target2)
        target3 = np.array(target3)
        target4 = np.array(target4)

        data1 = np.array(data1)
        data2 = np.array(data2)
        data3 = np.array(data3)
        data4 = np.array(data4)

        svc1 = LinearSVC(C=1.0, max_iter=50000)

        svc1.fit(data1, target1.T)

        vector1 = svc1.coef_[0]

        svc2 = LinearSVC(C=1.0, max_iter=50000)

        svc2.fit(data2, target2.T)

        vector2 = svc2.coef_[0]

        svc3 = LinearSVC(C=1.0, max_iter=50000)

        svc3.fit(data3, target3.T)

        vector3 = svc3.coef_[0]

        svc4 = LinearSVC(C=1.0, max_iter=50000)

        svc4.fit(data4, target4.T)

        vector4 = svc4.coef_[0]

        # loop for finding 6 cosines

        avg_cosine = 0

        avg_cosine += cosine(vector1, vector2) / 6

        avg_cosine += cosine(vector1, vector3) / 6

        avg_cosine += cosine(vector1, vector4) / 6

        avg_cosine += cosine(vector2, vector3) / 6

        avg_cosine += cosine(vector2, vector4) / 6

        avg_cosine += cosine(vector3, vector4) / 6

        plane_cosine.append(avg_cosine)

    plane_cosine.sort(reverse=True)

    return plane_cosine[0]


# abstraction index
# ratio of the average between group distance to the average within group distance


def abstraction_index(data, dichotomy=([1, 2, 3, 4], [5, 6, 7, 8])):

    labels1 = dichotomy
    label_1a = dichotomy[0]
    label_1b = dichotomy[1]

    indeces = []

    # labels1 = classification of interest, target
    # labels2 = subcategories

    for d in dichotomies:

        if d[0] != dichotomy[0] and d[0] != dichotomy[1]:

            data_1a = []  # even and high

            data_1b = []  # even and low

            data_2a = []  # odd and high

            data_2b = []  # odd and low

            label_2a = d[0]
            label_2b = d[1]

            target1 = []
            data1 = []

            target2 = []
            data2 = []

            index = 0
            for row in range(data.shape[0]):

                # if labels1[index] == label_1a: #if high

                if int(train_label[index]) in label_1a:

                    if int(train_label[index]) in label_2a:  # if even
                        data_1a.append(data[row])
                    else:
                        data_2a.append(data[row])  # if odd

                elif int(train_label[index]) in label_1b:  # if low

                    if int(train_label[index]) in label_2a:  # if even
                        data_1b.append(data[row])
                    else:
                        data_2b.append(data[row])  # if odd

                index += 1

            data_1a = np.array(data_1a)
            data_1b = np.array(data_1b)
            data_2a = np.array(data_2a)
            data_2b = np.array(data_2b)

            total_data = np.array([data_1a, data_1b, data_2a, data_2b])

            data_1a_mean = np.mean(data_1a)
            data_1b_mean = np.mean(data_1b)
            data_2a_mean = np.mean(data_2a)
            data_2b_mean = np.mean(data_2b)

            means = np.array([data_1a_mean, data_1b_mean, data_2a_mean, data_2b_mean])

            between_group = 0

            within_group = 0

            between_group += np.linalg.norm(data_1a_mean - data_1b_mean)
            between_group += np.linalg.norm(data_1a_mean - data_2b_mean)
            between_group += np.linalg.norm(data_2a_mean - data_2b_mean)
            between_group += np.linalg.norm(data_2a_mean - data_1b_mean)

            within_group += np.linalg.norm(data_1a_mean - data_2a_mean)
            within_group += np.linalg.norm(data_1b_mean - data_2b_mean)

            between_group /= 4
            within_group /= 2

            index = between_group / within_group

            indeces.append(index)

    indeces.sort(reverse=True)

    return indeces[0]


def error_bar_plot(
    NN_data, NN_sd, TME_data, TME_sd, random_data, random_sd, metric="CCGP"
):

    sample_CCGP = np.load(
        "/Users/anastasia/Desktop/randtensor/values/new_real_CCGP.npy"
    )
    sample_CCGP1 = np.sort(sample_CCGP)

    sample_CCGP1 = sample_CCGP1[::-1]
    random_CCGP = np.load("/Users/anastasia/Desktop/randtensor/values/random_CCGP.npy")

    random_CCGP = np.sort(random_CCGP)
    random_CCGP = random_CCGP[::-1]

    plt.scatter(range(0, len(sample_CCGP1)), sample_CCGP1, color="red")
    plt.scatter(range(0, len(CCGP_means)), CCGP_means, color="black")
    plt.errorbar(
        range(0, len(CCGP_means)), CCGP_means, yerr=2 * np.array(CCGP_sd), color="black"
    )

    plt.scatter(range(0, len(CFR_CCGP_means)), CFR_CCGP_means, color="orange")
    plt.errorbar(
        range(0, len(CFR_CCGP_means)),
        CFR_CCGP_means,
        yerr=2 * np.array(CFR_CCGP_sd),
        color="orange",
    )

    plt.scatter(range(0, len(random_CCGP)), random_CCGP, color="blue")

    plt.ylabel(metric, fontsize=30)
    plt.xlabel("Dichotomy Rank", fontsize=24)

    red_patch = mpatches.Patch(color="red", label="Actual")
    black_patch = mpatches.Patch(color="black", label="TME (T,N,C)")
    blue_patch = mpatches.Patch(color="blue", label="Isotropic")
    plt.legend(handles=[red_patch, black_patch, blue_patch], loc=(1.2, 0), fontsize=24)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.show()


def get_best_svm(data, dichotomy, labels):

    plane_cosine = []

    best_svc1 = 0
    best_svc2 = 0
    best_svc3 = 0
    best_svc4 = 0

    best_cosine = 0

    best_label1a = 0
    best_label2a = 0
    best_label3a = 0
    best_label4a = 0

    best_label1b = 0
    best_label2b = 0
    best_label3b = 0
    best_label4b = 0

    label1b = dichotomy[1][0]
    label2b = dichotomy[1][1]
    label3b = dichotomy[1][2]
    label4b = dichotomy[1][3]

    for perm in permutations(dichotomy[0]):

        label1a = perm[0]
        target1 = []
        data1 = []

        label2a = perm[1]
        target2 = []
        data2 = []

        label3a = perm[2]
        target3 = []
        data3 = []

        label4a = perm[3]
        target4 = []
        data4 = []

        for lab in range(len(labels)):
            if int(labels[lab]) == label1a:

                target1.append(0)

                data1.append(data[lab])

            if int(labels[lab]) == label1b:

                target1.append(1)

                data1.append(data[lab])

            if int(labels[lab]) == label2a:

                target2.append(0)

                data2.append(data[lab])

            if int(labels[lab]) == label2b:

                target2.append(1)

                data2.append(data[lab])

            if int(labels[lab]) == label3a:

                target3.append(0)

                data3.append(data[lab])

            if int(labels[lab]) == label3b:

                target3.append(1)

                data3.append(data[lab])

            if int(labels[lab]) == label4a:

                target4.append(0)

                data4.append(data[lab])

            if int(labels[lab]) == label4b:

                target4.append(1)

                data4.append(data[lab])

        target1 = np.array(target1)
        target2 = np.array(target2)
        target3 = np.array(target3)
        target4 = np.array(target4)

        data1 = np.array(data1)
        data2 = np.array(data2)
        data3 = np.array(data3)
        data4 = np.array(data4)

        svc1 = LinearSVC(C=1.0, max_iter=50000)

        svc1.fit(data1, target1.T)

        vector1 = svc1.coef_[0]

        svc2 = LinearSVC(C=1.0, max_iter=50000)

        svc2.fit(data2, target2.T)

        vector2 = svc2.coef_[0]

        svc3 = LinearSVC(C=1.0, max_iter=50000)

        svc3.fit(data3, target3.T)

        vector3 = svc3.coef_[0]

        svc4 = LinearSVC(C=1.0, max_iter=50000)

        svc4.fit(data4, target4.T)

        vector4 = svc4.coef_[0]

        avg_cosine = 0

        avg_cosine += cosine(vector1, vector2) / 6

        avg_cosine += cosine(vector1, vector3) / 6

        avg_cosine += cosine(vector1, vector4) / 6

        avg_cosine += cosine(vector2, vector3) / 6

        avg_cosine += cosine(vector2, vector4) / 6

        avg_cosine += cosine(vector3, vector4) / 6

        plane_cosine.append(avg_cosine)

        if avg_cosine > best_cosine:

            best_cosine = avg_cosine
            best_svc1 = svc1
            best_svc2 = svc2
            best_svc3 = svc3
            best_svc4 = svc4

            best_label1a = label1a
            best_label2a = label2a
            best_label3a = label3a
            best_label4a = label4a

            best_label1b = label1b
            best_label2b = label2b
            best_label3b = label3b
            best_label4b = label4b

    print(best_label1a, best_label2a, best_label3a, best_label4a)
    print(best_label1b, best_label2b, best_label3b, best_label4b)

    plane_cosine.sort(reverse=True)

    print(plane_cosine[0])

    return best_svc1, best_svc2, best_svc3, best_svc4


def get_projection_matrix(data):

    cor_mat1 = np.corrcoef(data_reshaped.T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    M = np.hstack(
        (
            eig_pairs[0][1].reshape(100, 1),
            eig_pairs[1][1].reshape(100, 1),
            eig_pairs[2][1].reshape(100, 1),
        )
    )

    return M


def calculate_plane(svm, M, num_points=4):

    new_points = []

    for a in range(num_points):

        point = np.ones(100)
        b = svm.intercept_[0]

        for x in range(99):

            param = svm.coef_[0][x]

            value = np.random.normal(0, 2)

            point[x] = value

            b -= value * param

        point[99] = b / (svm.coef_[0][99])

        new_points.append(point)

    points_3D = []

    for y in new_points:

        p = y.dot(M)

        points_3D.append(p)

    # calculate angle
    p1 = points_3D[0]
    p2 = points_3D[1]
    p3 = points_3D[2]

    v1 = p2 - p1
    v2 = p3 - p1

    coef = np.cross(v1, v2)

    a = coef[0]
    b = coef[1]
    c = coef[2]

    d = p1[0] * a + p1[1] * b + p1[2] * c

    # hypothesis = x_*(-a/c) + y_*(-b/c) + d/c

    return (-a / c), (-b / c), d / c


def plot_hyperplane(data, dichotomy, labels):

    # http://www.3leafnodes.com/plotly-getting-started
    # https://stackoverflow.com/questions/51558687/python-matplotlib-how-do-i-plot-a-plane-from-equation

    best_svc1, best_svc2, best_svc3, best_svc4 = get_best_svm(data, dichotomy, labels)

    title = "Actual Data (PS = 0.736)"

    M = get_projection_matrix(data)

    # True to color each digits distinctly, False to color by dichotomy
    col = True

    pca = PCA(n_components=3)

    pca_data = pca.fit_transform(data)

    colors = np.ones(len(labels), dtype=object)
    index = 0
    for label in new_labels:
        lab = int(label)

        if col == True:

            if lab == 1:
                color = "lightcoral"
            if lab == 2:
                color = "brown"
            if lab == 3:
                color = "maroon"
            if lab == 4:
                color = "red"
            if lab == 5:
                color = "cornflowerblue"
            if lab == 6:
                color = "royalblue"
            if lab == 7:
                color = "darkblue"
            if lab == 8:
                color = "lightsteelblue"
        else:
            if lab in dichotomy[0]:
                color = "red"
            if lab in dichotomy[1]:
                color = "blue"
        colors[index] = color
        index += 1

    X = pca_data

    x_min = math.floor(X[:, 0].min())
    x_max = math.ceil(X[:, 0].max())
    y_min = math.floor(X[:, 1].min())
    y_max = math.ceil(X[:, 1].max())

    x_, y_ = np.meshgrid(range(x_min, x_max), range(y_min, y_max))

    coef = calculate_plane(best_svc1, M)

    hypothesis = x_ * (coef[0]) + y_ * (coef[1]) + coef[2]

    coef = calculate_plane(best_svc2, M)

    hypothesis2 = x_ * (coef[0]) + y_ * (coef[1]) + coef[2]

    coef = calculate_plane(best_svc3, M)

    hypothesis3 = x_ * (coef[0]) + y_ * (coef[1]) + coef[2]

    coef = calculate_plane(best_svc4, M)

    hypothesis4 = x_ * (coef[0]) + y_ * (coef[1]) + coef[2]

    trace1 = go.Scatter3d(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        z=pca_data[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.95,
        ),
    )

    trace2 = go.Surface(
        x=x_,
        y=y_,
        z=hypothesis,
        name="Plane of Best Fit",
        opacity=0.7,
        colorscale="Greys",
        showscale=False,
    )

    trace3 = go.Surface(
        x=x_,
        y=y_,
        z=hypothesis2,
        name="Plane of Best Fit",
        opacity=0.7,
        colorscale="Reds",
        showscale=False,
    )

    trace4 = go.Surface(
        x=x_,
        y=y_,
        z=hypothesis3,
        name="Plane of Best Fit",
        opacity=0.7,
        colorscale="Blues",
        showscale=False,
    )

    trace5 = go.Surface(
        x=x_,
        y=y_,
        z=hypothesis4,
        name="Plane of Best Fit",
        opacity=0.7,
        colorscale="Greens",
        showscale=False,
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-10, 10], title="PCA1"),
            yaxis=dict(nticks=4, range=[-10, 10], title="PCA2"),
            zaxis=dict(nticks=4, range=[-10, 10], title="PCA3"),
        )
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)


def get_PS(data, labels):

    dichotomies = get_dichotomies()

    index = 0

    PS_ranks = np.ones((len(data), 35))

    for i in range(len(data)):

        PS = []

        for d in dichotomies:

            p_score_value = p_score(dataset, dichotomy=d, labels=labels)
            PS.append(p_score_value)

        # PS.sort(reverse = True)
        PS_ranks[index] = np.array(PS)

        index += 1

    return PS_ranks


def get_CCGP(data, labels):

    dichotomies = get_dichotomies()

    index = 0

    CCGP_ranks = np.ones((len(data), 35))

    for i in range(len(data)):

        CCGP = []

        for d in dichotomies:

            ccgp_value = ccgp(dataset, dichotomy=d, labels=labels)
            CCGP.append(ccgp_value)

        # CCGP.sort(reverse = True)
        CCGP_ranks[index] = np.array(CCGP)

        index += 1

    return PS_ranks


def get_mean_SD(values):

    x = values.shape[0]
    y = values.shape[1]

    means = np.ones(y)

    sd = np.ones(y)

    for a in range(y):
        means = []
        for b in range(x):
            means.append(values[b][a])

        means[a] = np.mean(means_data)
        sd[a] = np.std(means_data)

    return means, sd


def sort_means_sd(means, sd):

    values_dict = {}

    for x in range(len(means)):

        values_dict[x] = means[x]

    sorted_dict = sorted(values_dict.items(), key=operator.itemgetter(1), reverse=True)

    sorted_means = []

    sorted_sd = []

    for tup in sorted_dict:

        sorted_means.append(tup[1])

        sorted_sd.append(sd[tup[0]])

    return sorted_means, sorted_sd
