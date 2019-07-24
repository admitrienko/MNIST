import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Flatten, Embedding, Dense, Activation, Input
import gzip
import random
from utils import *

random.seed(1)

T = 5420
N = 100
C = 8
image_size = 784


def create_train_sample(train_data, train_labels):
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

    get_layer_output1 = K.function([model.layers[0].input], [model.layers[1].output])
    layer1_output = get_layer_output1([train_image_sample])[0]

    get_layer_output2 = K.function([model.layers[0].input], [model.layers[2].output])

    layer2_output = get_layer_output2([train_image_sample])[0]

    return layer1_output, layer2_output


def read_MNIST_data(
    train_image_path, test_image_path, train_label_path, test_label_path
):

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
    train_data = data.reshape(60000, np.sqrt(image_size), np.sqrt(image_size), 1)

    data = np.frombuffer(test_image, dtype=np.uint8).astype(np.float32)
    data = data[16:47040016]
    test_data = data.reshape(10000, np.sqrt(image_size), np.sqrt(image_size), 1)

    # image = np.asarray(train_data[235]).squeeze()
    # plt.imshow(image)
    # plt.show()

    return


def train_model(
    train_image_sample_random,
    parity_train_sample_random,
    magnitude_train_sample_random,
    epochs=400,
):

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
        epochs,
    )

    return model


if __name__ == "__main__":

    train_image_path = "./values/MNIST_train_images.gz"
    test_image_path = "./values/MNIST_test_images.gz"

    train_label_path = "./values/MNIST_train_labels.gz"
    test_label_path = "./values/MNIST_test_labels.gz"

    # create and shuffle training dataset, including flattened images and parity/magnitude labels

    train_data, train_labels, test_data, test_labels = create_train_sample(
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
