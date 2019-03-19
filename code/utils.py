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
    return np.random.normal(mu, std, N)

def compute_probability_scalar_gaussian_data(x, mu, var):
    # x can be a row vector or a scalar
    # Computes P(x), where P is N(mu,var)
    assert x.ndim == 1 or isinstance(x, int)

    std = np.sqrt(var)
    return scipy.stats.norm(mu,std).pdf(x)


class UnivariateGaussian():
    # Univariate
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def compute_probability(self, x):
        # x is a column vector
        # Computes P(x), where P is N(mu,var)
        std = np.sqrt(self.var)
        return scipy.stats.norm(self.mu,std).pdf(x)

    def sample(self, N):
        return np.random.normal(mu, std, N)


class UniformDataDistribution():
    # Univariate
    def __init__(self, total_outcomes, outcomes=None):
        if outcomes is not None:
            assert outcomes.shape[0] == total_outcomes

        self.total_outcomes = total_outcomes
        self.outcomes = outcomes

    def compute_probability(self, x):
        # Computes the probability of a sample from a uniform input data distribution
        return 1. / self.total_outcomes

    def sample(self, N):
        ind = np.random.randint(self.total_outcomes, size=N)
        return self.outcomes[ind,:]


class GaussianMixtureDataDistribution():
    def __init__(self, n_components, means, variances, mixture_probs):
        assert n_components == means.shape[0]
        assert n_components == variances.shape[0]
        assert n_components == mixture_probs.shape[0] and mixture_probs.ndim == 1
        assert means.shape[1] == variances.shape[1] == variances.shape[2]

        self.n_components = n_components
        self.means = means
        self.variances = variances
        self.mixture_probs = mixture_probs
        self.dimension = self.means.shape[1]

    def compute_probability(self, x):
        prob = 0.0
        for i in range(self.n_components):
            m = self.means[i,:]
            v = self.variances[i,:]
            prob = prob + scipy.stats.multivariate_normal.pdf(x, mean=m, cov=v) * self.mixture_probs[i]
        return prob

    def sample(self, N):
        component_idx = np.random.choice(
            self.n_components,
            size=N,
            replace=True,
            p=self.mixture_probs
        )
        X = [scipy.stats.multivariate_normal.rvs(self.means[i,:], self.variances[i,:,:]) for i in component_idx]
        X = np.array(X).reshape(N,self.dimension)

        return X


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

