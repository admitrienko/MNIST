import sys
sys.path.append('../')
from utils import *

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

def test_cosine():

    inputs = [[-1,0,0],[0,0,1],[1,0,0]]

    v1 = [1,0,0]
    
    outputs = [-1,0,1]

    for i in range(len(inputs)):
        assert prime_factors(v1,inputs[i]) == outputs[i]

    
'''
def test_learning_rates():

    mag_accuracies = []
    parity_accuracies = []

    lrs = [0.01, 0.005, 0.001, 0.0005, 0.00001, 0.00005]

    for rate in lrs:
    
        model1 = Model()

        inputs = Input(shape=(784,))

        layer1 = Dense(100, activation='tanh')(inputs)
        layer2 = Dense(100, activation='tanh')(layer1)

        parity_output = Dense(2, activation='softmax',name='parity_output')(layer2)
        magnitude_output = Dense(2, activation='softmax', name='magnitude_output')(layer2)


        model1 = Model(inputs=inputs, outputs=[parity_output, magnitude_output])


        model1.compile(keras.optimizers.Adam(lr=rate), loss={'parity_output': 'categorical_crossentropy', 'magnitude_output': 'categorical_crossentropy'})


    
        model1.fit(train_image_sample, [parity_train_sample, magnitude_train_sample], epochs=400)
    
        predictions = model1.predict(test_image_sample)
    
        accuracy = get_accuracy(predictions)
    
        mag_accuracies.append(accuracy[0])
    
        parity_accuracies.append(accuracy[1])
       

    for acc in mag_accuracies:
        assert(acc > 0.98)

    for acc in parity_accuracies:
        assert(acc > 0.98)    
'''    
        

def test_CCGP():

    toy_data_x = np.ones(200)
    toy_data_y = np.ones(200)
    toy_labels = np.ones(200)

    for i in range(100):
    
        toy_data_x[i] = np.random.normal(4, 0.5)
        toy_data_y[i] = np.random.normal(4, 0.5)
    

    for i in range(100,200):
    
        toy_data_x[i] = np.random.normal(-4, 0.5)
        toy_data_y[i] = np.random.normal(-4, 0.5)
    
    index = 0
    for x in range(1,9):
        for y in range(25):
        
            toy_labels[index] = x
        
            index += 1
        
    toy_data = np.ones((2,200))

    toy_data[0] = toy_data_x
    toy_data[1] = toy_data_y

    toy_data = toy_data.T

    ccgp_value = ccgp(toy_data, dichotomy = ([1,2,3,4], [5,6,7,8]),labels = toy_labels)

    assert(ccgp_value > 0.99)


    for i in range(200):
    
        toy_data_x[i] = np.random.normal(0, 5)
        toy_data_y[i] = np.random.normal(0, 5)
        
    toy_data = toy_data.T

    toy_data[0] = toy_data_x
    toy_data[1] = toy_data_y

    toy_data = toy_data.T

    ccgp_value = ccgp(toy_data, dichotomy = ([1,2,3,4], [5,6,7,8]),labels = toy_labels)

    assert(ccgp_value < 0.6)



def test_cov():

    inputs = np.array([[[0,0],[0,0]],[[1,1],[2,2]]])
    
    outputs = []

    outputs.append(np.array([[ 0.,  0.],
                            [ 0., 10.]]))
    outputs.append(np.array([[2., 4.],
                             [4., 8.]]))
    outputs.append(np.array([[ 5.,  5.],
                            [ 5., 5.]]))

    cov = cov_3D(inputs)
    for i in range(3):
        assert np.array_equal(cov[i],outputs[i])
    


def test_mean_center():

    NN_data = np.load('./values/new_NN_data.npy')
    NN_mean = np.load('./values/new_mean_2D.npy')
    
    data = NN_data - NN_mean

    T = data.shape[0]
    N = data.shape[1]
    C = data.shape[2]
    
    dimensions = [(N,C), (T,C), (T,N)]

    for dim in dimensions:
        
        total_sum = 0
        current_value = 0

        dim1 = dim[0]
        dim2 = dim[1]
        
        for x in range(dim1):

            for y in range(dim2):

                if dim == (N,C):
                    current_value = data[:,x, y]

                elif dim == (T,C):

                    current_value = data[x,:, y]

                elif dim == (T,N):

                    current_value = data[x,y,:]

                    
                total_sum += current_value

        total_mean = total_sum/(dim1*dim2)

        assert(np.mean(total_mean) < 1e-15)


def test_p_score():

    toy_data_x = np.ones(200)
    toy_data_y = np.ones(200)
    toy_labels = np.ones(200)

    for i in range(100):
    
        toy_data_x[i] = np.random.normal(4, 0.5)
        toy_data_y[i] = np.random.normal(4, 0.5)
    

    for i in range(100,200):
    
        toy_data_x[i] = np.random.normal(-4, 0.5)
        toy_data_y[i] = np.random.normal(-4, 0.5)
    
    index = 0
    for x in range(1,9):
        for y in range(25):
        
            toy_labels[index] = x
        
            index += 1
        
    toy_data = np.ones((2,200))

    toy_data[0] = toy_data_x
    toy_data[1] = toy_data_y

    toy_data = toy_data.T

    p_score = p_score(toy_data, dichotomy = ([1,2,3,4], [5,6,7,8]),labels = toy_labels)

    assert(p_score > 0.99)


    for i in range(200):
    
        toy_data_x[i] = np.random.normal(0, 5)
        toy_data_y[i] = np.random.normal(0, 5)

    toy_data[0] = toy_data_x
    toy_data[1] = toy_data_y

    toy_data = toy_data.T

    p_score = p_score(toy_data, dichotomy = ([1,2,3,4], [5,6,7,8]),labels = toy_labels)

    assert(p_score < 0.01)


def test_surrogate_primary_features():


    data = np.ones((100,50,25))
    
    for x in range(100):
        for y in range(50):
            for z in range(25):
                data[x,y,z] = np.random.normal(100,25)
                
    sigma_T, sigma_N, sigma_C = cov_3D(data)
    

    T = len(sigma_T)
    N = len(sigma_N)
    C = len(sigma_C)

    mean_T = np.zeros((T,T))
    mean_N = np.zeros((N,N))
    mean_C = np.zeros((C,C))


    for surr in range(len(data)):

        T_cov, N_cov, C_cov = cov_3D(surr)
    
        mean_T = mean_T + T_cov
            
        mean_N = mean_N + N_cov
    
        mean_C = mean_C + C_cov
    
    
    mean_T = mean_T/len(data)
    mean_N = mean_N/len(data)
    mean_C = mean_C/len(data) 
    
    #find percent difference of each element
    percent_diff_T = (100*abs(mean_T-sigma_T))/sigma_T
    percent_diff_N = (100*abs(mean_N-sigma_N))/sigma_N
    percent_diff_C = (100*abs(mean_C-sigma_C))/sigma_C

    #compare percent differences to make sure they are relatively constant
    T_flatten = percent_diff_T.flatten()
    N_flatten = percent_diff_N.flatten()
    C_flatten = percent_diff_C.flatten()

    T_percent_diff_1 = T_flatten[0]
    N_percent_diff_1 = N_flatten[0]
    C_percent_diff_1 = C_flatten[0]

    for T_val in T_flatten:
    
        percent_diff = (abs(T_val-T_percent_diff_1))/T_percent_diff_1
    
        assert(percent_diff < 0.20)

    for N_val in N_flatten:
    
        percent_diff = (abs(N_val-N_percent_diff_1))/N_percent_diff_1
    
        assert(percent_diff < 0.20)
    
    for C_val in C_flatten:
    
        percent_diff = (abs(C_val-C_percent_diff_1))/C_percent_diff_1
    
        assert(percent_diff < 0.20)




#def test_plane_projection():

    





if __name__ == "__main__":
    
    
    test_cosine()
    #test_learning_rates()
    test_CCGP()
    test_cov()
    test_mean_center()
    test_p_score()
    test_surrogate_primary_features()
    
