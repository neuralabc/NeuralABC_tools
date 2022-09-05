import numpy as np



class EigenGame():
    def __init__(self, n_components, epochs=100, learning_rate=0.1):
        
        """
        EigenGame implements the "EigenGame" algorithm, developed by
        Gemp et al. in 2020
        
        Constructor parameters:
        ----------------
        :param data: (2D array), required: a Numpy array containing the data to run PCA on, in the form (features, samples) 
        :param n_components: (int), required: the number of principal components to extract
        :param epochs: (int), optional: the number of iterations to calculate each eigenvector. Default = 100
        :param learning_rate: (float), optional: Learning rate of the algorithm. Default = 0.1

        Returns:
        ----------------
        None

        References:
        ----------------
        "EigenGame: PCA as a Nash Equilibrium"; Gemp et al., 2020 
        """
        
        self.n_components = n_components
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def fit_transform(self, data):
        """
        fit_transform() performs PCA on input data using the "EigenGame" algorithm, developed by
        Gemp et al. in 2020

        Parameters:
        ----------------
        None
        
        Returns:
        ----------------
        vectors
        :returns vectors: the principal components, stacked horizontally

        References:
        ----------------
        "EigenGame: PCA as a Nash Equilibrium"; Gemp et al., 2020 
        """
        self.data = data

        n_components = self.n_components
        learning_rate = self.learning_rate
        epochs = self.epochs
        
        dim = data.shape[1]
        vectors = np.ones((dim, n_components))
        for t in range(n_components):
            for epoch in range(epochs):
                rewards = np.dot(data, vectors[:, t])
                penalties = calc_penalties(data, vectors, t)

                delta_v = 2*np.dot(data.T, rewards - penalties)
                vectors[:, t] = vectors[:, t] + learning_rate * delta_v
                vectors[:, t] = vectors[:, t] / np.linalg.norm(vectors[:, t])

        self.eigenvectors = vectors.T
        return vectors.T
    
    def get_explained_variance_ratio(self):
        explained_variance_ratios = []
        
        cov_matrix_firstrow = np.zeros((self.data.shape[1]))
        for i in range(self.data.shape[1]):
            cov_matrix_firstrow[i] = np.dot(self.data[:,0], self.data[:,i])
        
        
        for v in self.eigenvectors:
            explained_variance_ratios.append(np.dot(v, cov_matrix_firstrow) / v[0])
        
        covariance_matrix_trace = 0
        for row in self.data:
            covariance_matrix_trace += np.dot(row, row)
        
        return explained_variance_ratios / covariance_matrix_trace


def calc_penalties(data, vectors, index):
    """
    calc_penalties is a helper function used by eigengame() and should never be called outside
    of it. 
    
    Parameters:
    ----------------
    :param data: (2D array), required: the data array for which we want to run PCA on.
    :param vectors: (2D array), required: the collection of eigenvectors that are being calculated by eigengame()
    :param index: (int), required: the index of the vector within vectors that we want to calculate the penalty for
    
    Returns:
    ----------------
    :returns penalties: The penalties array, which will be used by eigengame() to calculate the vector update
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
    :param data: (2D array), required: a Numpy array containing the data to run PCA on, in the form (features, samples) 
    :param n_components: (int), required: the number of principal components to extract
    :param epochs: (int), optional: the number of iterations to calculate each eigenvector. Default = 100
    :param learning_rate: (float), optional: Learning rate of the algorithm. Default = 0.1
    
    Returns:
    ----------------
    vectors
    :returns vectors: the principal components, stacked horizontally
    
    References:
    ----------------
    "EigenGame: PCA as a Nash Equilibrium"; Gemp et al., 2020 
    """
    dim = data.shape[1]
    vectors = np.ones((dim, n_components))
    for t in range(n_components):
        for epoch in range(epochs):
            rewards = np.dot(data, vectors[:, t])
            penalties = calc_penalties(data, vectors, t)
            
            delta_v = 2*np.dot(data.T, rewards - penalties)
            vectors[:, t] = vectors[:, t] + learning_rate * delta_v
            vectors[:, t] = vectors[:, t] / np.linalg.norm(vectors[:, t])
            
    return vectors.T
