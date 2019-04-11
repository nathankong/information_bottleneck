import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

class MonteCarloIntegrator():
    def __init__(self, f, dom, N, d):
        """
            Performs Monte Carlo integration to integrate functions over
            possibly multiple dimensions. The domain of integration assumes
            that the range for integration for each variable does not depend
            on other variables. i.e. the range of integration is defined by
            two integers.
    
            f:   Function that takes a numpy matrix as an argument and
                 returns a scalar value, where the rows are the variables
                 and the columns are the dimensions of the variables. i.e.
                 if $f$ is a function of $d$ scalar values, the argument
                 would be a numpy column vector of dimensions (d,1).
            dom: Numpy array of ranges to integrate over for each variable.
                 Each row corresponds a variable that corresponds to the
                 respective argument in the function. (i.e. first row would
                 be associated with the first entry of the argument, second
                 row with the second entry of the argument, etc.)
            N:   Number of MC samples to use
            d:   Number of variables in the integration
        """
        assert d == dom.shape[0]
        assert dom.shape[1] == 2 # Range of values

        self.f = f
        self.dom = dom
        self.N = N
        self.d = d

    def integrate(self):
        samples = self._sample_from_domain()
        func_values = self._compute_function_values(samples)
        assert func_values.shape[1] == 1, "Function must map to scalars."

        return 1./self.N * np.sum(func_values)

    def _sample_from_domain(self):
        # Uniformly sample points in the domain of integration
        samples = np.zeros((self.d,self.N))
        for i in range(self.d):
            samples[i,:] = np.random.uniform(self.dom[i,0], self.dom[i,1], self.N)
        return samples.T

    def _compute_function_values(self, samples):
        # Returns a numpy array of dimensions (N,1).
        return self.f(samples)


if __name__ == "__main__":

    def integrate_single_variable_func():
        # Function of one variable
        def f(x):
            return np.square(x)
    
        num_trials = 1000
        max_samples = np.log10(3000)
        Ms = np.zeros((num_trials,))
        Ns = np.logspace(1,max_samples,num_trials)
        d = 1
        dom = np.array([[0,1]])
    
        numerical_int, _ = integrate.quad(f, 0, 1)
        for i in range(num_trials):
            N = int(Ns[i])
            m = MonteCarloIntegrator(f, dom, N, d)
            val = m.integrate()
            Ms[i] = val
    
        plt.figure()
        #plt.plot(Ns, np.square(Ms - numerical_int))
        plt.plot(Ns, Ms - numerical_int)
        plt.xlabel("Number of samples")
        #plt.ylabel("Squared Error")
        plt.ylabel("Absolute Error")
        plt.savefig("temp.png")

    def integrate_double_variable_func():
        # Function of two variables
        def f(x):
            assert x.shape[1] == 2
            return (x[:,0] + x[:,1]).reshape(x.shape[0],1)

        num_trials = 1000
        max_samples = np.log10(3000)
        Ms = np.zeros((num_trials,))
        Ns = np.logspace(1,max_samples,num_trials)
        d = 2
        dom = np.array([[0,1],[0,1]])

        import time
        s = time.time()
        numerical_int, _ = integrate.nquad(lambda x, y: x + y, [[0,1],[0,1]])
        print "Numerical integration value: {}".format(numerical_int)
        print "Time for numerical integration: {}".format(time.time() - s)

        for i in range(num_trials):
            N = int(Ns[i])
            m = MonteCarloIntegrator(f, dom, N, d)
            val = m.integrate()
            Ms[i] = val

        m = MonteCarloIntegrator(f, dom, int(Ns[-1]), d)
        s = time.time()
        print "Monte Carlo integration value: {}".format(m.integrate())
        print "Time for Monte Carlo integration: {}".format(time.time() - s)

        plt.figure()
        #plt.plot(Ns, np.square(Ms - numerical_int))
        plt.plot(Ns, Ms - numerical_int)
        plt.xlabel("Number of samples")
        #plt.ylabel("Squared Error")
        plt.ylabel("Absolute Error")
        plt.savefig("temp.png")

    np.random.seed(10)
    #integrate_single_variable_func()
    integrate_double_variable_func()

    


