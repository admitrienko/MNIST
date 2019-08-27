import operator
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
import matplotlib.patches as mpatches

from MNIST_Abstraction_Testing.utils import *
from MNIST_Abstraction_Testing.pipeline_utils import *


if __name__ == "__main__":

    train_image_path = "./values/MNIST_train_images.gz"
    test_image_path = "./values/MNIST_test_images.gz"

    train_label_path = "./values/MNIST_train_labels.gz"
    test_label_path = "./values/MNIST_test_labels.gz"

    
    train_data, test_data, train_labels, test_labels = read_MNIST_data(train_image_path, test_image_path, train_label_path, test_label_path)
     

    train_image_sample, parity_train_sample, magnitude_train_sample, train_label = create_train_sample(
        train_data, train_labels
    )   

    train_image_sample_random, train_label_random, parity_train_sample_random, magnitude_train_sample_random, index_locations = shuffle_train_data(
        train_image_sample, parity_train_sample, magnitude_train_sample, train_label
    )

    model = train_model(
        train_image_sample_random,
        parity_train_sample_random,
        magnitude_train_sample_random,
    )

    layer1_output, layer2_output = hidden_layer_output(model, train_image_sample_random)

    plot_MDS(layer1_output, train_label_random, title="Layer 1 MDS")

    plot_MDS(layer2_output, train_label_random, title="Layer 2 MDS")

    # predictions = model.predict(test_image_sample)
    # accuracy = get_accuracy(predictions)

    # un-shuffle layer 2 output data
    layer2_output_new = np.ones((T*C, N))
    train_labels_new = np.ones(T*C)

    counter = 0
    for index in index_locations:
        index1 = int(index)

        train_labels_new[index1] = train_label_random[counter]
        layer2_output_new[index1] = layer2_output[counter]

        counter += 1


    import_randtensor("/Users/anastasia/Desktop/randtensor")

    #model = keras.models.load_model("./values/model1.h5")

    #old_NN_data = np.load("./values/old_NN_data.npy")

    #old_NN_labels = np.load("./values/old_labels.npy")

    old_NN_data = layer2_output_new

    old_NN_labels = train_labels_new

    NN_data, NN_labels = reduce_sample_size(old_NN_data, old_NN_labels)

    NN_mean_centered, NN_mean_centered_reshaped = mean_center(NN_data)

    TME_data = TME_sample(NN_mean_centered, NN_mean_centered_reshaped, num_samples = 3)



    #########
    
    #calculate PS and CCGP for real data


    NN_PS_ranks = get_PS([NN_data], NN_labels)
    NN_CCGP_ranks = get_CCGP([NN_data], NN_labels)

    #calculate PS and CCGP for TME data

    TME_PS_ranks = get_PS(TME_data, NN_labels)
    TME_CCGP_ranks = get_CCGP(TME_data, NN_labels)


    #calculate PS and CCGP for random data
    
    random_data = sample_random_data_2D(16000,100)

    random_PS_ranks = get_PS(random_data, NN_labels)
    random_CCGP_ranks = get_CCGP(random_data, NN_labels)

    
    #calculate sorted mean and SD

    NN_PS_means, NN_PS_sd = sort(get_mean_SD(NN_PS_ranks))
    NN_CCGP_means, NN_CCGP_sd = sort(get_mean_SD(NN_CCGP_ranks))

    TME_PS_means, TME_PS_sd = sort(get_mean_SD(TME_PS_ranks))
    TME_CCGP_means, TME_CCGP_sd = sort(get_mean_SD(TME_CCGP_ranks))

    random_PS_means, random_PS_sd = sort(get_mean_SD(random_PS_ranks))
    random_CCGP_means, random_CCGP_sd = sort(get_mean_SD(random_CCGP_ranks))    

    #plot PS

    error_bar_plot(NN_PS_means, NN_PS_sd, TME_PS_means, TME_PS_sd, random_PS_means, random_PS_sd)

    #plot CCGP
    
    error_bar_plot(NN_CCGP_means, NN_CCGP_sd, TME_CCGP_means, TME_CCGP_sd, random_CCGP_means, random_CCGP_sd)

    
    #create hyperplane plots for NN and TME data

    dichotomy = ([1,2,3,4], [5,6,7,8])

    plot_hyperplane(NN_data[0], dichotomy, NN_labels)

    plot_hyperplane(TME_data[0], dichotomy, NN_labels)
