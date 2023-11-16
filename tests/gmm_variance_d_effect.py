'''
This script can be used to generate the data required for figure 7.
    To make figure 7 just change d and re-run.
'''

import json
import numpy as np
import scipy as sp
import scipy.stats
import sys
sys.path.append('../')

from testing import variance_test
from entropic_maps import GaussianMixture

filename = "YOUR_FILE_NAME"
 
d = 5
    
nrange = [64, 128, 256, 512]
krange = [1]
trials = 100
n_mc = 500
epsilon = 5.0

mu_modes = 10
nu_modes = 15

mu_means = sp.stats.multivariate_normal.rvs(size=(mu_modes, d)) / np.sqrt(d)
nu_means = sp.stats.multivariate_normal.rvs(size=(nu_modes, d)) / np.sqrt(d)
nu_means[:,0] += 2.0

mu_weights = sp.stats.dirichlet(np.ones(mu_modes)).rvs()
nu_weights = sp.stats.dirichlet(np.ones(nu_modes)).rvs()

mu_covs = [np.eye(d) * np.random.rand() / d for i in range(mu_modes)]
nu_covs = [np.eye(d) * np.random.rand() / d for i in range(nu_modes)]

map_generator = GaussianMixture(mu_means, mu_covs, mu_weights, nu_means, nu_covs, nu_weights)

results = {}

for nsamples in nrange:
    for k in krange:
        results[str((nsamples, k))] = variance_test(map_generator, 
            epsilon, 
            n_mc, 
            nsamples, 
            trials, 
            k=k
        ).tolist()


dataset = {
    "results": results,
    "parameters": {
        "nrange": nrange,
        "krange": krange,
        "trials": trials,
        "n_mc": n_mc,
        "d":d,
        "epsilon": epsilon,
        "mu_means": [m.tolist() for m in mu_means],
        "nu_means": [m.tolist() for m in nu_means],
        "mu_covs": [c.tolist() for c in mu_covs],
        "nu_covs": [c.tolist() for c in nu_covs],
        "mu_weights": mu_weights.tolist(),
        "nu_weights": nu_weights.tolist()
    }
}

store_path = filename + ".json"
store_file = open(store_path, "w")
store_file.write(json.dumps(dataset))
store_file.close()