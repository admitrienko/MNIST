import numpy as np

def reshape1(data):

    data_reshaped = np.reshape(data, (2,2,2), order = 'F') 

    return data_reshaped

def test_reshape1():

    input0 = np.array([[1,2,3,4],[5,6,7,8]])

    output0 = np.array([[1,2],[3,4]])

    output1 = np.array([[5,6],[7,8]])
    
    reshaped = reshape1(input0)
    
    reshaped0 = reshaped[0]
    reshaped1 = reshaped[1]

    assert reshaped0[0] == output1[0]
    assert reshaped0[1] == output1[1]
    assert reshaped0[2] == output1[2]
    assert reshaped0[3] == output1[3]
    
    assert reshaped1[0] == output2[0]
    assert reshaped1[1] == output2[1]
    assert reshaped1[2] == output2[2]
    assert reshaped1[3] == output2[3]
    
