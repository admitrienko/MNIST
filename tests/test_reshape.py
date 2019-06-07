import numpy as np

def reshape1(data):

    data_reshaped = np.reshape(data, (4,2,3), order = 'F')

    return data_reshaped

def test_reshape1(input1):

    reshaped = reshape1(input1):

    

    assert np.array_equal(reshaped[:,0], input1[0:4])

    assert np.array_equal(reshaped[:,1], input1[4:8])
    
if __name__ == "__main__":
    data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]]) #8x3
    test_reshape1(data)
    #test_swap_axes(data)
