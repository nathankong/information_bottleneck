import torch
import numpy as np
import scipy.integrate as integrate

class MutualInformationEstimator():
    def __init__(self, model, device, noise_distribution, data_distribution, input_dataset):
        # samples: (N,)
        # noise_distribution is a function that computes the probability of a sample (outputs a scalar)
        # data_distribution is a function that computes the probability of an input sample (outputs a scalar)
        # TODO: We are assuming the input dataset consists of discrete scalar values (so it is a numpy array
        # of possible input values)

        # TODO: For the time being, we are assuming that the input dataset are scalar values
        # and is fed in as a column vector
        assert input_dataset.shape[1] == 1 and input_dataset.ndim == 2

        self.model = model
        self.device = device
        self.noise_distribution = noise_distribution
        self.data_distribution = data_distribution
        self.input_dataset = input_dataset
        self.tol = 1e-7

    def compute_mutual_information(self, layer_samples, num_samples_per_outcome):
        uncond_entropy = self.compute_unconditional_entropy(layer_samples)

        mutual_information = uncond_entropy
        for i in range(self.input_dataset.shape[0]):
            curr_outcome = self.input_dataset[i]
            cond_entr = self.compute_conditional_entropy(curr_outcome, num_samples_per_outcome)
            print cond_entr
            cond_entropy = self.data_distribution(curr_outcome) * cond_entr
            mutual_information -= cond_entropy

        return mutual_information

    def compute_conditional_entropy(self, sample, num_samples_per_outcome):
        # Compute outputs for sample from different noise realizations
        inputs = np.tile(sample, (num_samples_per_outcome,1))
        model_outputs_noise, _ = self.model(torch.from_numpy(inputs).float().to(self.device))
        model_outputs_noise = model_outputs_noise.detach().cpu().numpy()

        # estimated_conditional_distribution is the estimated distribution for 
        # p_{T_\ell | X = x_i} = p_{S_\ell | X = x_i} \ast \phi \approx \hat{p}_{S_\ell}^{(i)} \ast \phi
        estimated_conditional_distribution = lambda x: (1./num_samples_per_outcome)* \
                                                np.sum(self.noise_distribution(x - model_outputs_noise))
        integrand = lambda x: estimated_conditional_distribution(x) * np.log(1./estimated_conditional_distribution(x) + self.tol)

        # SP estimator: h(p_{T_\ell | X = x_i}) \approx h(\hat{p}_{S_{\ell}^{(i)}} \ast \phi)
        conditional_entropy, _ = integrate.quad(integrand, -np.inf, np.inf)

        return conditional_entropy

    def compute_unconditional_entropy(self, samples):
        # estimated_distribution is the estimated distribution for
        # p_{T_\ell} = p_{S_\ell} \ast \phi \approx \hat{p}_{S_\ell} \ast \phi

        # TODO: For the time being, we are assuming that the hidden dimension is 1
        assert samples.shape[1] == 1

        num_samples = samples.shape[0]

        # TODO: Since we are assuming that hidden dimension is 1, 'x' is a scalar here
        estimated_distribution = lambda x: (1./num_samples)*np.sum(self.noise_distribution(x - samples))

        # Integrand for computing the entropy. i.e. f(x) log (1/f(x))
        integrand = lambda x: estimated_distribution(x) * np.log(1./estimated_distribution(x) + self.tol)

        # SP estimator: h(p_{T_\ell}) \approx h(\hat{p}_{S_\ell} \ast \phi)
        unconditional_entropy, _ = integrate.quad(integrand, -np.inf, np.inf)

        return unconditional_entropy


