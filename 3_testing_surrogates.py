import math
from itertools import combinations 
from itertools import permutations
from sklearn import manifold
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import *
            


def sample_random_data_2D(dimensions):
    
    random_data = np.ones((dimensions))

    for i in range(dimensions[0]):
        for j in range(dimensions[0]):
            random_data[i,j] = np.random.normal()

    return random_data



if __name__ == "__main__":


    #calculate PS and CCGP for real data

    NN_data = np.load("./values/new_NN_data.npy")

    NN_labels = np.load("./values/new_labels.npy")

    NN_PS_ranks = get_PS(NN_data, NN_labels)
    NN_CCGP_ranks = get_CCGP(NN_data, NN_labels)

    #calculate PS and CCGP for TME data

    TME_data = np.load("./values/surrogates.npy")

    TME_PS_ranks = get_PS(TME_data, NN_labels)
    TME_CCGP_ranks = get_CCGP(CCGP_data, NN_labels)


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

    




