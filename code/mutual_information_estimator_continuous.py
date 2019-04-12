from multiprocessing import Pool
import torch
import numpy as np
import scipy.integrate as integrate

import dill

from mc_integration import MonteCarloIntegrator

class MutualInformationEstimator():
    # Note that the multiprocessing only works if things are run on the CPU
    def __init__(self, model, noise_distribution, data_distribution):
        # TODO: We are assuming the input dataset consists of discrete scalar values (so it is a numpy array
        # of possible input values)
        # samples: (N,)
        # noise_distribution is a function that computes the probability of a sample (outputs a scalar)
        # data_distribution is a function that computes the probability of an input sample (outputs a scalar)

        self.model = model
        self.noise_distribution = noise_distribution
        self.data_distribution = data_distribution
        self.tol = 1e-10

    def compute_mutual_information(self, layer_samples, num_samples_per_outcome, layer_name, num_mc_samples):
        uncond_entropy = self.compute_unconditional_entropy(layer_samples)
    
        mutual_information = uncond_entropy
        samples = self.data_distribution.sample(num_mc_samples) # Samples num_mc_samples from distribution
    
        # TODO: We assume, for the time being, that the samples are one dimensional so that
        # samples.shape == (num_samples,)
        p = Pool(processes=10)
        jobs = list()
        for i in range(num_mc_samples):
            arg = [samples[i], num_samples_per_outcome, layer_name]
            job = apply_async(p, self.compute_conditional_entropy, arg)
            jobs.append(job)

        results = list()
        for job in jobs:
            results.append(job.get())
        p.close()
        p.join()

        cond_entropy = 1. / num_mc_samples * np.sum(results)
        mutual_information -= cond_entropy

        return mutual_information

    def compute_unconditional_entropy(self, samples):
        # estimated_distribution is the estimated distribution for
        # p_{T_\ell} = p_{S_\ell} \ast \phi \approx \hat{p}_{S_\ell} \ast \phi
    
        # TODO: For the time being, we are assuming that the hidden dimension is 1
        assert samples.shape[1] == 1
    
        num_samples = samples.shape[0]
    
        # Using the Monte Carlo Integrator
        num_mc_integration_samples = 1000
        def estimated_distribution(x):
            # TODO: We are still assuming the hidden layer dimensionality is 1 (see the reshape operation)
            samples_t = np.tile(samples.reshape((num_samples,)), (num_mc_integration_samples,1))
            return (1./num_samples)*np.sum(self.noise_distribution.compute_probability(x - samples_t), axis=1)

        def integrand(x):
            # These operations are all elementwise.
            res = -1. * estimated_distribution(x) * np.log(estimated_distribution(x) + self.tol)
            return res.reshape(res.shape[0], 1)

        dom = np.array([[-10,10]])
        m_integrator = MonteCarloIntegrator(integrand, dom, num_mc_integration_samples, 1)
        unconditional_entropy = m_integrator.integrate()

        return unconditional_entropy
    
    def compute_conditional_entropy(self, sample, num_samples_per_outcome, layer_name):
        device = torch.device("cpu")

        # Compute outputs for sample from different noise realizations
        assert sample.shape[0] == 1
        inputs = np.tile(sample, (num_samples_per_outcome,1))
        _, model_outputs = self.model(torch.from_numpy(inputs).float().to(device))
        model_outputs = model_outputs[layer_name].detach().cpu().numpy()

        # estimated_conditional_distribution is the estimated distribution for 
        # p_{T_\ell | X = x_i} = p_{S_\ell | X = x_i} \ast \phi \approx \hat{p}_{S_\ell}^{(i)} \ast \phi

        num_mc_integration_samples = 1000
        def estimated_conditional_distribution(x):
            # TODO: We are still assuming the hidden layer dimensionality is 1 (see the reshape operation)
            model_outputs_t = np.tile(model_outputs.reshape((num_samples_per_outcome,)), (num_mc_integration_samples,1))
            return (1./num_samples_per_outcome)*np.sum(self.noise_distribution.compute_probability(x - model_outputs_t), axis=1)

        def integrand(x):
            res = -1. * estimated_conditional_distribution(x) * np.log(estimated_conditional_distribution(x) + self.tol)
            return res.reshape(res.shape[0],1)

        dom = np.array([[-10,10]])
        m_integrator = MonteCarloIntegrator(integrand, dom, num_mc_integration_samples, 1)
        conditional_entropy = m_integrator.integrate()

        return conditional_entropy

def apply_async(pool, func, args):
    payload = dill.dumps((func, args))
    return pool.apply_async(run_dill_encoded, (payload,))

def run_dill_encoded(payload):
    func, args = dill.loads(payload)
    return func(*args)

