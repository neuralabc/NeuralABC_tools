import numpy as np

def calc_penalties(data, vectors, index):
    """
    calc_penalties is a helper function used by eigengame() and should never be called outside
    of it. 
    
    Parameters:
    ----------------
    data: the data array for which we want to run PCA on
    vectors: the collection of eigenvectors that are being calculated by eigengame()
    index: the index of the vector within vectors that we want to calculate the penalty for
    
    Returns:
    ----------------
    The penalties array, which will be used by eigengame() to calculate the vector update

    References:
    ----------------
    "EigenGame: PCA as a Nash Equilibrium"; Gemp et al., 2020 
    """
    vec = vectors[:, index]
    penalties = np.zeros_like(np.dot(data, vectors[:, 0]))
    
    for i in range(index):
        result = np.dot(data, vectors[:, i])
        penalties += (np.dot(np.dot(data, vec), result) /
         np.dot(result, result)
        ) * result

    return penalties 
     
def eigengame(data, n_components, epochs=100, learning_rate=0.1):
    """
    eigengame() performs PCA on input data using the "EigenGame" algorithm, developed by
    Gemp et al. in 2020
    
    Parameters:
    ----------------
    data: a Numpy array containing the data to run PCA on
    
    n_components: the number of principal components to extract
    
    epochs: the number of iterations to calculate each eigenvector. Default = 100
    
    learning_rate: Learning rate of the algorithm. Default = 0.1
    
    Returns:
    ----------------
    M, vectors
    Where M is the covariance matrix and vectors is a Numpy array containing 
    the principal components, stacked horizontally
    
    References:
    ----------------
    "EigenGame: PCA as a Nash Equilibrium"; Gemp et al., 2020 
    """
    M = np.dot(data.T, data)
    dim = M.shape[0]
    vectors = np.ones((dim, n_components))
    for t in range(n_components):
        for epoch in range(epochs):
            rewards = np.dot(data, vectors[:, t])
            penalties = calc_penalties(data, vectors, t)
            
            delta_v = 2*np.dot(data.T, rewards - penalties)
            vectors[:, t] = vectors[:, t] + learning_rate * delta_v
            vectors[:, t] = vectors[:, t] / np.linalg.norm(vectors[:, t])
            
    return M, vectors.T
