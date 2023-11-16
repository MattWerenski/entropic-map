import numpy as np    

def variance_test(map_generator, epsilon, n_mc, nsamples, trials, k=1):
    '''
    variance_test - empirically estimates the variance of the map estimator
        across many trials

    :param map_generator: An object which can perform the map estimation
    :param epsilon: regularization paramter
    :param n_mc: number of monte carlo samples to use in each trial
    :param nsamples: number of samples to draw from each measure
    :param trials: number of trials to perform
    :param k: number of batches to use when estimating the map
        for the variance test it makes the most sense to use k=1
    :return: an array where each entry is the variance estimate from a single trial
    '''

    if epsilon <= 0.0:
        raise Exception("epsilon must be positive")
    
    if type(n_mc) != int or n_mc < 1:
        raise Exception("n_mc must be a positive integer")
    
    if type(nsamples) != int or nsamples < 1:
        raise Exception("nsamples must be a positive integer")
    
    if type(trials) != int or k < 1:
        raise Exception("trials must be a positive integer")
    
    if type(k) != int or k < 1:
        raise Exception("k must be a positive integer")

    variance_estimates = np.zeros(trials)
    for i in range(trials):
        variance_estimates[i] = variance_single_test(map_generator, epsilon, n_mc, nsamples, k=k)
    
    return variance_estimates

def variance_single_test(map_generator, epsilon, n_mc, nsamples, k=1):
    '''
    variance_single_test - empirically estimates the variance of the map estimator

    :param map_generator: An object which can perform the map estimation
    :param epsilon: regularization paramter
    :param n_mc: number of monte carlo samples to use
    :param nsamples: number of samples to draw from each measure
    :param k: number of batches to use when estimating the map
        for the variance test it makes the most sense to use k=1
    :return: the variance estimate from a single trial
    '''
    
    if epsilon <= 0.0:
        raise Exception("epsilon must be positive")
    
    if type(n_mc) != int or n_mc < 1:
        raise Exception("n_mc must be a positive integer")
    
    if type(nsamples) != int or nsamples < 1:
        raise Exception("nsamples must be a positive integer")
    
    if type(k) != int or k < 1:
        raise Exception("k must be a positive integer")

    # handles sample allocation of nsamples into k batches, with rounding as needed
    m_no_round = int(np.floor(nsamples / k))
    batch_samples = [m_no_round] * k
    remaining = int(nsamples - m_no_round * k)
    for i in range(remaining):
        batch_samples[i] += 1
        
    # samples that the map will be evaluated on
    evaluate_samples = map_generator.sample_marginals(n_mc)[0]

    # captures two indpendent images
    images1 = np.zeros(evaluate_samples.shape)
    images2 = np.zeros(evaluate_samples.shape)
    for batch in range(k):
        batch_size = batch_samples[batch]
    
        images1 += (1/k) * map_generator.estimate_map(epsilon, batch_size, evaluate_samples)
        images2 += (1/k) * map_generator.estiamte_map(epsilon, batch_size, evaluate_samples)

    variance_estimate = np.power(images1 - images2, 2).sum() / (2 * n_mc)
    return variance_estimate

def exact_test(map_generator, epsilon, n_mc, nsamples, trials, k=1):
    '''
    exact_test - empirically estimates the error of the map estimator
        across many trials (compared against the exact known map)

    :param map_generator: An object which can perform the map estimation
    :param epsilon: regularization paramter
    :param n_mc: number of monte carlo samples to use in each trial
    :param nsamples: number of samples to draw from each measure
    :param trials: number of trials to perform
    :param k: number of batches to use when estimating the map
        for the exact test any k is fine, but k=1 is empirically the best
    :return: an array where each entry is the error estimate from a single trial
    '''

    if epsilon <= 0.0:
        raise Exception("epsilon must be positive")
    
    if type(n_mc) != int or n_mc < 1:
        raise Exception("n_mc must be a positive integer")
    
    if type(nsamples) != int or nsamples < 1:
        raise Exception("nsamples must be a positive integer")
    
    if type(trials) != int or k < 1:
        raise Exception("trials must be a positive integer")
    
    if type(k) != int or k < 1:
        raise Exception("k must be a positive integer")

    exact_estimates = np.zeros(trials)
    for i in range(trials):
        exact_estimates[i] = exact_single_test(map_generator, epsilon, n_mc, nsamples, k=k)
    
    return exact_estimates

def exact_single_test(map_generator, epsilon, n_mc, nsamples, k=1):
    '''
    exact_test - empirically estimates the error of the map estimator
        (compared against the exact known map)

    :param map_generator: An object which can perform the map estimation
    :param epsilon: regularization paramter
    :param n_mc: number of monte carlo samples to use
    :param nsamples: number of samples to draw from each measure
    :param k: number of batches to use when estimating the map
        for the exact test any k is fine, but k=1 is empirically the best
    :return: the error estimate from a single trial
    '''

    if epsilon <= 0.0:
        raise Exception("epsilon must be positive")
    
    if type(n_mc) != int or n_mc < 1:
        raise Exception("n_mc must be a positive integer")
    
    if type(nsamples) != int or nsamples < 1:
        raise Exception("nsamples must be a positive integer")
    
    if type(k) != int or k < 1:
        raise Exception("k must be a positive integer")

    # handles sample allocation of nsamples into k batches, with rounding as needed
    m_no_round = int(np.floor(nsamples / k))
    batch_samples = [m_no_round] * k
    remaining = int(nsamples - m_no_round * k)
    for i in range(remaining):
        batch_samples[i] += 1
        
    # samples that the map will be evaluated on
    evaluate_samples = map_generator.sample_marginals(n_mc)[0]
    images = np.zeros(evaluate_samples.shape)
    
    for batch in range(k):
        batch_size = batch_samples[batch]
        images += (1/k) * map_generator.estimate_map(epsilon, batch_size, evaluate_samples)
            
    exacts = map_generator.exact_map(epsilon, evaluate_samples)
    
    error_estimate = np.power(images - exacts, 2).sum() / n_mc
    return error_estimate