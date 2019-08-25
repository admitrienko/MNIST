import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Flatten, Embedding, Dense, Activation, Input
import gzip
import random

from MNIST_Abstraction_Testing.utils import *


def create_train_sample(train_data, train_labels):
    
    """Creates a training sample with equal numbers (T) of each digit (1-8)
    
	    # Arguments 
	        train_data (2D array): training data size (60000, 28, 28, 1)
	        train_labels (1D array): digit labels for training data length 60000
    
        # Returns 
            train_image_sample: new train image sample with compressed images (size 784)
            parity_train_sample: one-hot vector representing parity of each image in new sample
            magnitude_train_sample: one-hot vector representing magnitude of each image in new sample
            train_label: array with labels for each digit
    
    """
    
    train_image_sample = np.ones((T*C, image_size))
    parity_train_sample = np.ones((2, T*C))
    magnitude_train_sample = np.ones((2, T*C))
    train_label = np.ones(T*C)

    counts = [0, 0, 0, 0, 0, 0, 0, 0]

    sample_size = 0

    for current_number in range(1, 9):

        counter = 0

        while counter < len(train_labels):
            current_label = train_labels[counter]

            if current_label == current_number:

                train_image_sample[sample_size] = np.reshape(
                    train_data[counter], (1, image_size)
                )
                train_label[sample_size] = train_labels[counter]

                if train_labels[counter] < 5:
                    magnitude_train_sample[0][sample_size] = 0

                else:
                    magnitude_train_sample[1][sample_size] = 0

                if train_labels[counter] % 2 == 0:
                    parity_train_sample[0][sample_size] = 0

                else:
                    parity_train_sample[1][sample_size] = 0

                sample_size += 1
                counts[current_number - 1] += 1

            if counts[current_number - 1] == T:
                break

            counter += 1

    return train_image_sample, parity_train_sample, magnitude_train_sample, train_label


def shuffle_train_data(
    train_image_sample, parity_train_sample, magnitude_train_sample, train_label
):
    
    """Shuffle the training data before inputting to neural network
	    # Arguments 
            train_image_sample, parity_train_sample, magnitude_train_sample, train_label (array)
    
        # Returns 
            train_image_sample_random,
            train_label_random,
            parity_train_sample_random,
            magnitude_train_sample_random (array): shuffled versions of input data
            
            index_locations (array) = indeces for old locations of the shuffled data
    
    """
    
    train_image_sample_random = np.ones((T*C, image_size))
    train_label_random = np.ones(T*C)
    parity_train_sample_random = np.ones((2, T*C))
    magnitude_train_sample_random = np.ones((2, T*C))

    index = range(0, T*C)
    index_locations = np.ones(T*C)
    counter = 0
    while len(index) > 0:

        random_int = np.random.randint(0, len(index))
        random = index[random_int]
        train_image_sample_random[counter] = train_image_sample[random]
        train_label_random[counter] = train_label[random]

        parity_train_sample_random[:, counter] = parity_train_sample[:, random]
        magnitude_train_sample_random[:, counter] = magnitude_train_sample[:, random]

        index = np.delete(index, random_int)

        index_locations[counter] = random
        counter += 1

    return (
        train_image_sample_random,
        train_label_random,
        parity_train_sample_random,
        magnitude_train_sample_random,
        index_locations,
    )


def hidden_layer_output(model, train_image_sample):
    
    
    """Get output of 2 hidden layers in neural network
    
	    # Arguments 
	        model (keras.engine.training.Model): trained Keras neural network model
	        train_image_sample (array): Array containing images sample to input to model
        # Returns 
            layer1_output (array): (T*C, N) shape array of layer 1 output
            layer2_output (array): (T*C, N) shape array of layer 2 output
    
    """

    get_layer_output1 = K.function([model.layers[0].input], [model.layers[1].output])
    layer1_output = get_layer_output1([train_image_sample])[0]

    get_layer_output2 = K.function([model.layers[0].input], [model.layers[2].output])

    layer2_output = get_layer_output2([train_image_sample])[0]

    return layer1_output, layer2_output


def read_MNIST_data(
    train_image_path, test_image_path, train_label_path, test_label_path
):
    
    """Get output of 2 hidden layers in neural network
    
	    # Arguments 
	        train_image_path, test_image_path, train_label_path, test_label_path (string):
            paths to MNIST datasets (.gz format) for training/test images & labels
            
        # Returns 
            train_data, test_data, train_labels, test_labels (array): Arrays with values
    
    """

    with gzip.open(train_image_path, "rb") as f:
        train_image = f.read()
    f.close()

    with gzip.open(test_image_path, "rb") as f:
        test_image = f.read()
    f.close()

    with gzip.open(train_label_path, "rb") as f:
        train_label = f.read()
    f.close()

    with gzip.open(test_label_path, "rb") as f:
        test_label = f.read()
    f.close()

    parity = np.zeros(2).reshape(1, -1)
    magnitude = np.zeros(2).reshape(1, -1)

    f = gzip.open(train_label_path, "r")
    f.read(8)
    buf = f.read(60000)
    train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    f.close()

    f = gzip.open(test_label_path, "r")
    f.read(8)
    buf = f.read(10000)
    test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    f.close()

    data = np.frombuffer(train_image, dtype=np.uint8).astype(np.float32)
    data = data[16:47040016]
    train_data = data.reshape(60000, int(np.sqrt(image_size)), int(np.sqrt(image_size)), 1)

    data = np.frombuffer(test_image, dtype=np.uint8).astype(np.float32)
    data = data[16:47040016]
    test_data = data.reshape(10000, int(np.sqrt(image_size)), int(np.sqrt(image_size)), 1)

    # image = np.asarray(train_data[235]).squeeze()
    # plt.imshow(image)
    # plt.show()

    return train_data, test_data, train_labels, test_labels


def train_model(
    train_image_sample_random,
    parity_train_sample_random,
    magnitude_train_sample_random,
    epochs=400,
):
    
    """Train the model
    
	    # Arguments 
	        train_image_sample_random:
            parity_train_sample_random:
            magnitude_train_sample_random:
            OPTIONAL
            epochs = number of training epochs
            
        # Returns 
            model (keras.engine.training.Model): trained Keras neural network model 
    
    """

    # compile and fit model

    model = Model()

    inputs = Input(shape=(image_size,))

    layer1 = Dense(100, activation="tanh")(inputs)
    layer2 = Dense(100, activation="tanh")(layer1)

    parity_output = Dense(2, activation="softmax", name="parity_output")(layer2)
    magnitude_output = Dense(2, activation="softmax", name="magnitude_output")(layer2)

    model = Model(inputs=inputs, outputs=[parity_output, magnitude_output])

    model.compile(
        keras.optimizers.Adam(lr=0.00005),
        loss={
            "parity_output": "categorical_crossentropy",
            "magnitude_output": "categorical_crossentropy",
        },
    )

    model.fit(
        train_image_sample_random,
        [parity_train_sample_random.T, magnitude_train_sample_random.T],
        epochs=epochs,
    )

    return model
    
    
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
    
def sample_random_data_2D(dimensions):
            
    """Creates an isoptropic random dataset
            
        # Arguments 
	dimensions (tuple): size of random dataset (T*C,N)
    
        # Returns 
            random_data (array): (T*C, N) shape random dataset
    
    """
    
    random_data = np.ones((dimensions))

    for i in range(dimensions[0]):
        for j in range(dimensions[0]):
            random_data[i,j] = np.random.normal()

    return random_data
