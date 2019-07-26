import numpy as np
import sys
import keras
from keras.models import Model, Sequential
from utils import *



def reduce_sample_size(data, labels, T=2000, C=8, N=100, old_T=5420):
    
    """Reduce sample size from 5420 to 2000 to make covariance computations feasible
	# Arguments 
		data (array): Neural network output data with two dimensional shape (old_T*C, N)
            	labels (array): Corresponding condition labels with same length as input data (N).
		
		OPTIONAL
		Input values of current T,N,C dimensions, as well as previous T dimension before reducing size.
    
        # Returns 
           	new_data: New data with two dimensional shape (T*C, N)
		new_labels: New labels, length N
    
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
    
    """Subtract the tensor marginal mean from neural network output
	 # Arguments 
	        data (array): Neural network output data with 3 dimensional shape (T,N,C)
    
        # Returns 
		mean_centered: NN data with tensor marginal mean subtracted away, shape (T,N,C)
		mean_reshaped: tensor marginal mean shape (T*C, N)
    
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
	
    mean_centered = np.reshape(mean_centered, (T, N ,C), order="F")

    return mean_centered, mean_reshaped


def TME_sample(mean_centered, mean_reshaped, num_samples):
    
    """Sample
	    # Arguments 
		mean_centered: neural network output data with tensor marginal mean subtracted away, shape (T,N,C)
		mean_reshaped: tensor marginal mean shape (T*C, N)
		num_samples (int): number of TME samples to generate
    
        # Returns 
            	surrogates_reshaped (array): Array of TME surrogate datasets with tensor marginal mean added back 
		and reshaped back to (T*C, N)
    
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


def import_randtensor(pathname):
	
    """Import the randtensor directory
    
	# Arguments 
		pathname (string): Path of randtensor directory for importing
    
        # Returns 
            	None
  
    """

    sys.path.append(pathname)
    import utils as u
    import randtensor as r

    return None




if __name__ == "__main__":

    import_randtensor("/Users/anastasia/Desktop/randtensor")

    model = keras.models.load_model("./values/model1.h5")

    old_NN_data = np.load("./values/old_NN_data.npy")

    old_NN_labels = np.load("./values/old_labels.npy")

    NN_data, NN_labels = reduce_sample_size(old_NN_data, old_NN_labels)

    NN_mean_centered = mean_center(NN_data)

    surrogate_data = TME_sample(NN_mean_centered)
