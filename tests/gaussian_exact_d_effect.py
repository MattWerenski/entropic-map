'''
This script can be used to generate the data required for figure 5.
    To make the complete plot you will need to re-run this with different d's
'''

import json
import numpy as np
import scipy as sp
import scipy.stats
import sys
sys.path.append('../')

from testing import exact_test
from entropic_maps import GaussianMeasure
 
filename = "YOUR_FILE_NAME"

d = 5
nrange = [64, 128, 256, 512]
krange = [1]
trials = 100
n_mc = 500
epsilon = 2.0

mu_mean = np.zeros(d)
nu_mean = np.zeros(d) 
nu_mean[0] = 2

mu_diag = (np.random.rand(d) * 0.9 + 0.1) / d
nu_diag = (np.random.rand(d) * 0.9 + 0.1) / d

mu_eigs = sp.stats.ortho_group.rvs(d)
nu_eigs = sp.stats.ortho_group.rvs(d)

mu_cov = mu_eigs @ np.diag(mu_diag) @ mu_eigs.T
nu_cov = nu_eigs @ np.diag(nu_diag) @ nu_eigs.T

map_generator = GaussianMeasure(
    np.array(mu_mean),
    np.array(mu_cov),
    np.array(nu_mean),
    np.array(nu_cov)
)

results = {}

for nsamples in nrange:
    for k in krange:
        results[str((nsamples, k))] = exact_test(map_generator, 
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
        "d":d,
        "n_mc": n_mc,
        "epsilon": epsilon,
        "mu_mean": mu_mean.tolist(),
        "nu_mean": nu_mean.tolist(),
        "mu_cov": mu_cov.tolist(),
        "nu_cov": nu_cov.tolist(),
    }
}

store_path = filename + ".json"
store_file = open(store_path, "w")
store_file.write(json.dumps(dataset))
store_file.close()