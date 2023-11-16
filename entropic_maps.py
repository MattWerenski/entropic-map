import numpy as np
import scipy as sp
import scipy.stats
from scipy.linalg import sqrtm
import ot

def coupling_to_map(coupling, target_support):
    '''
    coupling_to_map - converts a diffuse coupling into a map
        by computing the conditional expectation of each slice

    :param coupling: 2D numpy array with coupling[i,j] being the mass from Xi to Yj
    :param target_support: locations of the points Yj
    :return: 2D numpy array where the i'th row is the image of the i'th sample under the map
    '''

    if np.abs(np.sum(coupling) - 1.0) > 0.00001:
        raise Exception("coupling does not sum to 1")
        
    if np.min(coupling) < 0.0:
        raise Exception("coupling cannot have negative entries")
        
    if coupling.shape[1] != target_support.shape[0]:
        raise Exception("coupling.shape[1] must equal target_support.shape[0]")
    
    unnormalized_map = coupling @ target_support
    normalized_map = unnormalized_map / coupling.sum(1)[:,np.newaxis]
    
    return normalized_map


def MALA(density, nsamples, burn, step_size):
    '''
    MALA - Metropolis Adjusted Langevin Algorithm for sampling from a strongly
        log concave measure

    :param density: If the density of the measure is e^{-U} then density represents U
    :param nsamples: number of samples to generate
    :param burn: number of iterations to run before taking a sample
    :param step_size: parameter for MALA
    :return: nsamples drawn according to the given density
    '''

    if type(nsamples) != int:
        raise Exception("nsamples must be an int")
        
    if nsamples < 1:
        raise Exception("nsamples must be at least 1")
    
    if type(burn) != int:
        raise Exception("burn must be an int")
    
    if burn < 0:
        raise Exception("burn must ne non-negative")
    
    if step_size <= 0:
        raise Exception("step size must be positive")

    samples = sp.stats.norm.rvs(size=(density.dim, nsamples))
    scaler = np.sqrt(2 * step_size)    
    
    for _ in range(burn):
        
        offsets =  sp.stats.norm.rvs(size=(density.dim, nsamples)) * scaler
                
        means = samples - step_size * density.grad(samples)
                
        new_samples = means + offsets
                
        log_alphas = density.value(samples) - density.value(new_samples) 
        offset_1 = samples - new_samples + step_size * density.grad(new_samples)
        offset_2 = new_samples - samples + step_size * density.grad(samples)
        
        log_alphas -= np.sum(offset_1 * offset_2, 0) / (4 * step_size)
        log_alphas += np.sum(offset_2 * offset_2, 0) / (4 * step_size)
        
        log_us = np.log(np.random.rand(nsamples))
        
        samples = new_samples * (log_us <= log_alphas) + samples * (log_us > log_alphas)
        
        
    return samples


class SamplingMeasure:
    '''
    SamplingMeasure - A general class for working with empirical measures.
        This class only requires methods for generating samples
    '''
    
    def __init__(self, mu_sampler, nu_sampler):
        '''
        __init__ 

        :param mu_sampler: function which takes an integer and returns that many samples
            from the source measure
        :param nu_sampler: function which takes an integer and returns that many sampels
            from the target measure
        '''

        self.mu_sampler = mu_sampler
        self.nu_sampler = nu_sampler
            
    def estimate_map(self, epsilon, nsamples, evaluate_samples):
        '''
        estimate_map - Estimates the image of the given samples by sampling from the measures
        
        :param epsilon: regularization parameter
        :param nsamples: number of samples to draw from the source and target measure
        :param evaluate_samples: points to compute the estimated map at
        :return: the image of the samples under the empirical entropic map
        '''

        if epsilon <= 0.0:
            raise Exception("epsilon must be positive")
            
        if type(nsamples) != int:
            raise Exception("nsamples must be an int")
        
        if nsamples < 1:
            raise Exception("nsamples must be at least 1")
            
        mu_hat_support, nu_hat_support, mu_hat_mass, nu_hat_mass = self.sample_marginals(nsamples)
        images = np.zeros(evaluate_samples.shape)
        
        for i, e_sample in enumerate(evaluate_samples):
            mu_hat_support[0] = e_sample
            cost_matrix_with_sample = (1/2) * ot.utils.dist(mu_hat_support, nu_hat_support)
                
            # get the optimal couplings and turn it into a map
            coupling = ot.sinkhorn(mu_hat_mass, nu_hat_mass, cost_matrix_with_sample, epsilon)
            images[i] = coupling_to_map(coupling, nu_hat_support)[0]

        return images
        
    
    def sample_marginals(self, nsamples):
        '''
        sample_marginals - Draws n samples from the source and target measures
        :param nsamples: number of samples to draw
        :return: the support of the samples and the mass vector
        '''
        
        if type(nsamples) != int:
            raise Exception("nsamples must be an int")
        
        if nsamples < 1:
            raise Exception("nsamples must be at least 1")
        
        mu_support = self.mu_sampler(nsamples)
        nu_support = self.nu_sampler(nsamples)
        
        return mu_support, nu_support, np.ones(nsamples) / nsamples, np.ones(nsamples) / nsamples
    
    
class GaussianMeasure(SamplingMeasure):
    
    '''
    GaussianMeasure - A special case of a sampling measure where the samples are drawn from
        Gaussian measures. This map has an explicit form which can be called in exact_map
    '''
    
    def __init__(self, mu_mean, mu_cov, nu_mean, nu_cov):
        '''
        __init__

        :param mu_mean: mean of the source measure
        :param mu_cov: covariance of the source measure
        :param nu_mean: mean of the target measure
        :param nu_cov: covariance of the target measure
        '''

        if mu_cov.shape[0] != mu_cov.shape[1]:
            raise Exception("mu_cov must be a square matrix")

        if mu_mean.shape[0] != mu_cov.shape[0]:
            raise Exception("mu mean and covariance must be the same dimension")
        
        if nu_cov.shape[0] != nu_cov.shape[1]:
            raise Exception("nu_cov must be a square matrix")
        
        if nu_mean.shape[0] != nu_cov.shape[0]:
            raise Exception("nu mean and covariance must be the same dimension")
        
        if mu_mean.shape[0] != nu_mean.shape[0]:
            raise Exception("mu and nu must be the same dimension")

        self.d = mu_cov.shape[0]
        self.mu_mean = mu_mean
        self.mu_cov = mu_cov
        self.nu_mean = nu_mean
        self.nu_cov = nu_cov
        
        # sampling functions for SamplingMeasure parent class to use
        mu_sampler = lambda nsamples: sp.stats.multivariate_normal.rvs(mean=mu_mean, cov=mu_cov, size=nsamples)
        nu_sampler = lambda nsamples: sp.stats.multivariate_normal.rvs(mean=nu_mean, cov=nu_cov, size=nsamples)
        super().__init__(mu_sampler, nu_sampler)
                
    def exact_map(self, epsilon, evaluate_samples):
        '''
        exact_map - Computs the exact entropic map between the two gaussians. This is taken from
            Theroem 1 in "Entropic Optimal Transport between Unbalanced Gaussian Measures has a Closed Form"
        
        :param epsilon: regularization parameter
        :param evaluate_samples: points to find the image of under the entropic map
        :return: the image of evaluate_samples
        '''

        if epsilon <= 0:
            raise Exception("epsilon must be positive")
        
        sqrt_mu = sqrtm(self.mu_cov)
        D_eps = sqrtm(4 * sqrt_mu @ self.nu_cov @ sqrt_mu + (epsilon ** 2) * np.eye(self.d))
        C_eps = (1/2) * sqrt_mu @ D_eps @ np.linalg.inv(sqrt_mu) - (epsilon / 2) * np.eye(self.d)
        
        # C_eps is the off-diagonal block of the covariance of pi_eps and we can use the formula
        # for the conditional expectation of a m.v. gaussian given its first d coordinates (see wikipedia)
        image = C_eps @ np.linalg.inv(self.mu_cov) @ (evaluate_samples - self.mu_mean).T + self.nu_mean[:, np.newaxis]
        
        return image.T
    
    
class GaussianMixture(SamplingMeasure):
    '''
    GaussianMixture - Special case of a sampling measure where the source and target are
        both GMMs
    '''
    
    def __init__(self, mu_means, mu_covs, mu_weights, nu_means, nu_covs, nu_weights):
        '''
        __init__ 

        :param mu_means: list of means of the source components
        :param mu_covs: list of covariances of the source components
        :param mu_weights: probability of drawing a sample from each source component
        :param nu_means: list of means of the target components
        :param nu_covs: list of covariances of the target components
        :param nu_weights: probability of drawing a sample from each target component
        '''

        if len(mu_means) != len(mu_covs) or len(mu_means) != mu_weights.shape[0]:
            raise Exception("mu_means, mu_covs, and mu_weights must all have the same length")
        
        if len(nu_means) != len(nu_covs) or len(nu_means) != nu_weights.shape[0]:
            raise Exception("nu_means, nu_covs, and nu_weights must all have the same length")
        
        if np.sum(mu_weights) != 1.0 or np.min(mu_weights) < 0:
            raise Exception("mu_weights must sum to 1.0 and be non-negative")
        
        if np.sum(nu_weights) != 1.0 or np.min(nu_weights) < 0:
            raise Exception("nu_weights must sum to 1.0 and be non-negative")

        self.d = mu_means[0].shape[0]
        self.mu_means = mu_means
        self.mu_covs = mu_covs
        self.mu_weights = mu_weights
        self.nu_means = nu_means
        self.nu_covs = nu_covs
        self.nu_weights = nu_weights
        
        # sampling functions for SamplingMeasure parent class to use
        mu_sampler = lambda nsamples : self.sample(nsamples, mu_means, mu_covs, mu_weights)
        nu_sampler = lambda nsamples : self.sample(nsamples, nu_means, nu_covs, nu_weights)
        
        super().__init__(mu_sampler, nu_sampler)
        
        
    def sample(self, nsamples, means, covs, weights):
        '''
        sample - generates a set of samples from the GMM with given means
            covariances and weights

        :param nsamples: number of samples to generate
        :param means: list of means of each component
        :param covs: list of covariances of each component
        :param weights: probability of drawing a sample from each component
        :return: nsamples from the GMM
        '''

        if type(nsamples) != int:
            raise Exception("nsamples must be an int")
        
        if nsamples < 1:
            raise Exception("nsamples must be at least 1")
        
        if np.sum(weights) != 1.0 or np.min(weights) < 0:
            raise Exception("weights must sum to 1.0 and be non-negative")
        
        # determines which components to sample from
        uniform = np.random.rand(nsamples)
        weight_cumsum = np.cumsum(weights)
        thresholds = np.subtract.outer(weight_cumsum, uniform).T > 0.0
        indices = np.argmax(thresholds, axis=1)
        
        # actually draws the samples using the selected components
        samples = np.zeros((nsamples, self.d))
        for i in range(nsamples):
            
            mean = means[indices[i]]
            cov = covs[indices[i]]
            samples[i] = sp.stats.multivariate_normal.rvs(mean=mean, cov=cov)
            
        return samples
        
    
class PieceWise:
    '''
    PieceWise - density to draw from of the form q||x||^2 + max_i <x, a_i> + b_i
        where q, a_i and b_i are the parameters to choose.
    '''

    def __init__(self, quadratic_coeff, slopes, intercepts):
        '''
        __init__

        :param quadratic_coeff: coefficient of the norm-squared term
        :param slopes: list of slopes a_i for the maximum term 
        :param intercepts: list of the intercepts b_i for the maximum term
        '''

        if quadratic_coeff <= 0.0:
            raise Exception("quadratic_coeff must be positive")
        
        if len(slopes) != len (intercepts):
            raise Exception("slopes and intercepts must have the same number of terms")

        self.quadratic_coeff = quadratic_coeff
        self.slopes = np.array(slopes)
        self.intercepts = np.array(intercepts).reshape(-1,1)
        
    def value(self, X):
        '''
        value - computes the value of the density at the given points

        :param X: points at which the density is evaluated
        :return: array of the values of the density
        '''

        pw_term = (self.slopes @ X + self.intercepts).max(0)
        q_term = self.quadratic_coeff * (X * X).sum(0)
        return pw_term + q_term
    
    def grad(self, X):
        '''
        grad - computes the gradient of the density at the given point

        :param X: points at which the gradient is evaluated
        :return: array of the gradients of the density
        '''

        pw_term = self.slopes[(self.slopes @ X + self.intercepts).argmax(0)].T
        q_term = self.quadratic_coeff * 2 * X
        return pw_term + q_term
    
class LogConcave(SamplingMeasure):
    '''
    LogConcave - Special case of a sampling measure where the source and target are parameterized
        by log concave densities and sampled using MALA
    '''
    
    def __init__(self, mu_density, nu_density, burn=500, step_size=0.01):
        '''
        __init__

        :param mu_density: density of the source measure
        :param nu_density: density of the target measure
        :param burn: burn parameter for MALA
        :param step_size: step_size parameter for MALA
        '''

        self.mu_density = mu_density
        self.nu_density = nu_density
        self.burn = burn
        self.step_size = step_size
        
        # sampling functions for SamplingMeasure parent class to use
        mu_sampler = lambda nsamples : MALA(mu_density, nsamples, burn, step_size).T
        nu_sampler = lambda nsamples : MALA(nu_density, nsamples, burn, step_size).T
        
        super().__init__(mu_sampler, nu_sampler)        