import numpy as np

def EigenGame2(data, n):
    M = np.matmul(data.T, data)
    dim = M.shape[0]
    vecs = []
    gradients = []
    
    step = 0.01
    
    for i in range(n):
        vecs.append(np.random.rand(dim))
    for epoch in range(100000): #number of iterations
        gradients = []
        for index in range(len(vecs)):
            
            rewards = np.matmul(data, vecs[index])
            rewards = np.matmul(data.T, data) / dim

            rewards = np.matmul(rewards, vecs[index])
            
            sum = np.zeros(rewards.shape[0])
            for ancestor in range(index):
                
                coefficient = np.dot(np.matmul(data,vecs[index]), np.matmul(data, vecs[ancestor]))
                
                sum += coefficient * vecs[ancestor]
            
            sum = sum / dim
            gr = rewards - sum
            rgr = gr - np.dot(gr, vecs[index]) * vecs[index]
            gradients.append(step * rgr)
        
        if(epoch % 10000 == 0):
            step = step / 10

        for i in range(len(gradients)):
            vecs[i] = vecs[i] + gradients[i]
            vecs[i] = vecs[i] / (np.linalg.norm(vecs[i]))
    return vecs
