import numpy as np
import scipy.stats

def sample_gaussian(N, mu, var):
    # mu is column vector
    assert var.shape[0] == mu.shape[0]
    assert var.shape[0] == var.shape[1]
    mu = mu.reshape((mu.shape[0],))
    return np.random.multivariate_normal(mu, var, N)

def sample_univariate_gaussian(N, mu, var):
    # Generates N scalaras from N(mu,var)
    # mu and var are scalars
    std = np.sqrt(var)
    return np.random.normal(N, mu, std)

def compute_probability_scalar_gaussian_data(x, mu, var):
    # x can be a row vector or a scalar
    # Computes P(x), where P is N(mu,var)
    assert x.ndim == 1 or isinstance(x, int)

    std = np.sqrt(var)
    return scipy.stats.norm(mu,std).pdf(x)


class UnivariateGaussian():
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def compute_probability(self, x):
        # x is a column vector
        # Computes P(x), where P is N(mu,var)
        std = np.sqrt(self.var)
        return scipy.stats.norm(self.mu,std).pdf(x)


class UniformDataDistribution():
    def __init__(self, total_outcomes):
        self.total_outcomes = total_outcomes

    def compute_probability(self, x):
        # Computes the probability of a sample from a uniform input data distribution
        return 1. / self.total_outcomes


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

