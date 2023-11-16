'''
This script can be used to generate the data required for figure 3.
    To make figure 3 just change epsilon and re-run.
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

nrange = [64, 128, 192, 256, 320, 384, 448, 512]
krange = [1]
trials = 100
n_mc = 500
epsilon = 1.0

mu_means = [
    np.array([0.0,0.0,0.0,0.0,0.0]),
    np.array([1.0,-1.0,-1.0,-1.0,-1.0,]),
    np.array([-1.0,-1.0,-1.0,-1.0,1.0,]),
    np.array([0.0,0.0,3.0,0.0,0.0])
]
mu_covs = [
    np.eye(5) / 2,
    np.eye(5) / 5,
    np.diag([0.5,1.0,1.5,1.0,0.5]),
    np.diag([0.1,0.1,3.0,0.1,0.1])
]
mu_weights = np.ones(4) / 4

nu_means = [
    np.array([1.0,0.0,0.0,0.0,0.0]),
    np.array([0.0,1.0,0.0,0.0,0.0]),
    np.array([0.0,0.0,1.0,0.0,0.0]),
    np.array([0.0,0.0,0.0,1.0,0.0])
]
nu_covs = [
    np.eye(5),
    np.eye(5) / 2,
    np.eye(5) / 4,
    np.eye(5) / 8
]
nu_weights = np.array([0.1,0.2,0.3,0.4])
map_generator = GaussianMixture(mu_means, mu_covs, mu_weights, nu_means, nu_covs, nu_weights)

results = {}

for nsamples in nrange:
    for k in krange:
        results[str((nsamples, k))] = variance_test(
            map_generator, 
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