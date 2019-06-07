import numpy as np

def reshape1(data):

    data_reshaped = np.reshape(data, (4,2,3), order = 'F')

    return data_reshaped

def test_reshape1():
    
    original_data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]])
    
    new_data = reshape1(original_data)
    
    assert np.array_equal(new_data[:,0], original_data[0:4])

    assert np.array_equal(new_data[:,1], original_data[4:8])
    
def test_swap_axes():

    data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]])

    original_data = reshape1(data)

    swapped_data = np.swapaxes(original_data,1,2)

    assert np.array_equal(swapped_data[:,:,0], original_data[:,0])
    
    assert np.array_equal(swapped_data[:,:,1], original_data[:,1])
    
    
def first_moment(tensor):
    sizes = np.array(tensor.shape)
    nmodes = len(sizes)
    tensorIxs = range(nmodes)
    tensor0 = tensor
    for x in tensorIxs:
        nxSet = list(set(tensorIxs).difference(set([x])))
        mu = sumTensor(tensor0, nxSet)/np.prod(sizes[nxSet])
        tensor0 = tensor0 - mu
    return tensor-tensor0
        
def test_mean():
    
    data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]])

    original_data = reshape1(data)

    swapped_data = np.swapaxes(original_data,1,2)
    
    T = swapped_data.shape[0]
    N = swapped_data.shape[1]
    C = swapped_data.shape[2]
    
    mean = first_moment(swapped_data)
    
    total_sum = 0
    current_sum = 0

    X = data - mean

    #mean across T
    for n_value in range(N):
    
        for c_value in range(C):
        
            current_sum = X[:,n_value, c_value]
        
            total_sum += current_sum
    
        
    total_mean = total_sum/ (N*C)

    assert np.mean(total_mean) < 1e-7

#mean across N
    total_sum = 0
    current_sum = 0

    for t_value in range(T):
    
        for c_value in range(C):
        
            current_sum = X[t_value,:, c_value]
        
            total_sum += current_sum
        
        
    total_mean = total_sum/ (T*C)

    assert np.mean(total_mean) < 1e-7

#mean across C
    total_sum = 0
    current_sum = 0

    for t_value in range(T):
    
        for n_value in range(N):
        
            current_sum = X[t_value,n_value, :]
        
            total_sum += current_sum
        
        
    total_mean = total_sum/ (T*N)

    assert np.mean(total_mean) < 1e-7
        
        
    
    
if __name__ == "__main__":
    #data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12], [13,14,15],[16,17,18],[19,20,21],[22,23,24]]) #8x3
    
    test_reshape1()
    test_swap_axes()
    test_mean()
