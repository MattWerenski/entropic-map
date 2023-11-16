'''
This script can be used to generate the data required for figures 1 and 4
    To make figure 4 just change epsilon and re-run.
'''

import json
import numpy as np
import sys
sys.path.append('../')

from testing import exact_test
from entropic_maps import GaussianMeasure
 
filename = "YOUR_FILE_NAME"

nrange = [64, 128, 192, 256, 320, 384, 448, 512]
krange = [1, 2, 4, 8, 16, 32]
trials = 100
n_mc = 500
epsilon = 5.0

mu_mean = [0,0,0,0,0]
nu_mean = [2,1,0,-1,-2]

mu_cov = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]
nu_cov = [[2,0,0,0,0], [0,0.5,0,0,0], [0,0,1,0,0], [0,0,0,0.1,0], [0,0,0,0,5]]

map_generator = GaussianMeasure(
    np.array(mu_mean),
    np.array(mu_cov),
    np.array(nu_mean),
    np.array(nu_cov)
)

results = {}

for nsamples in nrange:
    for k in krange:
        results[str((nsamples, k))] = exact_test(
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
        "mu_mean": mu_mean,
        "nu_mean": nu_mean,
        "mu_cov": mu_cov,
        "nu_cov": nu_cov
    }
}

store_path = filename + ".json"
store_file = open(store_path, "w")
store_file.write(json.dumps(dataset))
store_file.close()