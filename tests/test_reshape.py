import numpy as np

def reshape1(data):

    data_reshaped = np.reshape(data, (2,2,2), order = 'F') 

    return data_reshaped

def test_reshape1():

    input1 = np.array([[1,2,3,4],[5,6,7,8]])

    output1 = np.array([[1,2],[3,4]])

    output2 = np.array([[5,6],[7,8]])

    assert reshape1(input1)[0] == output1
    assert reshape1(input1)[1] == output2
    
