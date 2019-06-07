import numpy as np

def reshape1(data):

    data_reshaped = np.reshape(data, (4,2,3), order = 'F')

    return data_reshaped

def test_reshape1():
    
    original_data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]])
    
    new_data = reshape1(original_data)
    
    assert np.array_equal(new_data[:,0], original_data[0:4])

    assert np.array_equal(new_data[:,1], original_data[4:8])
    
    
#def swap_axes1():
    
#def test_swap_axes(input1):
    
    
    
    
    
    
if __name__ == "__main__":
    data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]]) #8x3
    step1 = reshape1(data)
    test_reshape1()
    
    #step2 = swap_axes1(step1)
    #test_swap_axes(data)
