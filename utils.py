import numpy as np

def sample_gaussian(N, mu, var):
    # mu is column vector
    assert var.shape[0] == mu.shape[0]
    assert var.shape[0] == var.shape[1]
    mu = mu.reshape((mu.shape[0],))
    return np.random.multivariate_normal(mu, var, N)

if __name__ == "__main__":
    mu = np.array([[0]])
    cov = np.diag([0.1])
    print(mu.shape, cov.shape)
    
    samp = sample_gaussian(1000,mu,cov)
    print(np.mean(samp), np.std(samp)**2)

    mu = np.array([[0],[0]])
    cov = np.diag([0.1,0.1])
    samp = sample_gaussian(1000,mu,cov)
    print(np.mean(samp, axis=0), np.cov(samp.T))

