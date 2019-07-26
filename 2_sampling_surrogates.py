import numpy as np
import sys
import keras
from keras.models import Model, Sequential
from utils import *


def import_randtensor(pathname):

    sys.path.append(pathname)
    import utils as u
    import randtensor as r

    return None


# reduce sample size from 5420 to 2000 to make covariance computations feasible


def reduce_sample_size(data, labels, T=2000, C=8, N=100, old_T=5420):
    
    """Find classification accuracy of neural network trained to determine magnitude/parity
	    # Arguments 
	        predictions (array): output of model.predict(), Nx2 array
	        test_magnitudes (array): N-length array of magnitude labels
	        test_parities (array): N-length array of parity labels
    
        # Returns 
            mag_accuracy (double): accuracy of classifying magnitude
	        parity_accuracy (double): accuracy of classifying parity
    
    """

    new_data = np.ones(((T * C), N))
    new_NN_labels = np.ones(T * C)

    index = 0
    for x in range(C):
        for y in range(T):

            new_data[index] = data[x * old_T + y]
            new_labels[index] = old_NN_labels[x * old_T + y]

            index += 1

    return new_data, new_labels


def mean_center(data):
    
    """Find classification accuracy of neural network trained to determine magnitude/parity
	    # Arguments 
	        predictions (array): output of model.predict(), Nx2 array
	        test_magnitudes (array): N-length array of magnitude labels
	        test_parities (array): N-length array of parity labels
    
        # Returns 
            mag_accuracy (double): accuracy of classifying magnitude
	        parity_accuracy (double): accuracy of classifying parity
    
    """

    T = data.shape[0]
    N = data.shape[1]
    C = data.shape[2]

    # reshape from ((T*C),N) to (T,C,N)
    data_2D = np.reshape(data, (T, C, N), order="F")

    # reshape from (T,C,N) to (T,N,C)
    data_2D = np.swapaxes(data_2D, 1, 2)

    mean = u.first_moment(data_2D)

    mean_reshaped = np.reshape(mean, ((T * C), N), order="F")

    mean_centered = data_2D - mean_reshaped

    return mean_centered, mean_reshaped


def TME_sample(mean_centered, mean_reshaped, num_samples):
    
    """Find classification accuracy of neural network trained to determine magnitude/parity
	    # Arguments 
	        predictions (array): output of model.predict(), Nx2 array
	        test_magnitudes (array): N-length array of magnitude labels
	        test_parities (array): N-length array of parity labels
    
        # Returns 
            mag_accuracy (double): accuracy of classifying magnitude
	        parity_accuracy (double): accuracy of classifying parity
    
    """

    T = mean_centered.shape[0]
    N = mean_centered.shape[1]
    C = mean_centered.shape[2]

    sigma_T, sigma_N, sigma_C = cov_3D(mean_centered, T, N, C)

    sizes = (T, N, C)
    covs = [sigma_T, sigma_N, sigma_C]

    rand = r.randtensor(sizes)
    rand.fitMaxEntropy(covs)

    surrogates = rand.sampleTensors(num_samples)

    surrogates_reshaped = []

    # add back mean to surrogates
    for surr in surrogates:

        # reshape from (T,N,C) to ((T*C), N)
        surrogate_dataset = np.reshape(surr, ((T * C), N), order="F")

        surrogate_dataset = surrogate_dataset + mean_reshaped

        surrogates_reshaped.append(surrogate_dataset)

    return surrogates_reshaped


if __name__ == "__main__":

    import_randtensor("/Users/anastasia/Desktop/randtensor")

    model = keras.models.load_model("./values/model1.h5")

    old_NN_data = np.load("./values/old_NN_data.npy")

    old_NN_labels = np.load("./values/old_labels.npy")

    NN_data, NN_labels = reduce_sample_size(old_NN_data, old_NN_labels)

    NN_mean_centered = mean_center(NN_data)

    surrogate_data = TME_sample(NN_mean_centered)
