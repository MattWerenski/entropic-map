'''
This script can be used to generate the data required for figure 2.
    To make figure 2 just change epsilon and re-run.
'''

import json
import scipy as sp
import scipy.stats
import sys
sys.path.append('../')

from testing import variance_test
from entropic_maps import LogConcave, PieceWise
 
filename = "YOUR_FILE_NAME"

nrange = [64, 128, 192, 256, 320, 384, 448, 512]
krange = [1]
trials = 100
n_mc = 500
epsilon = 1.0
burn = 500
step_size = 0.01

mu_quadratic_coeff = 1.0
nu_quadratic_coeff = 0.75

mu_slopes = sp.stats.norm.rvs(size=(20, 5))
nu_slopes = sp.stats.norm.rvs(size=(20, 5))

mu_intercepts = sp.stats.norm.rvs(size=(20))
nu_intercepts = sp.stats.norm.rvs(size=(20)) 

mu_density = PieceWise(mu_quadratic_coeff, mu_slopes, mu_intercepts)
nu_density = PieceWise(nu_quadratic_coeff, nu_slopes, nu_intercepts)

map_generator = LogConcave(mu_density, nu_density, burn=burn, step_size=step_size)

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
        "mu_quadratic_coefficient": mu_quadratic_coeff,
        "nu_quadratic_coefficient": nu_quadratic_coeff,
        "mu_slopes": mu_slopes.tolist(),
        "nu_slopes": nu_slopes.tolist(),
        "mu_intercepts": mu_intercepts.tolist(),
        "nu_intercepts": nu_intercepts.tolist(),
        "burn": burn,
        "step_size": step_size
    }
}

store_path = filename + ".json"
store_file = open(store_path, "w")
store_file.write(json.dumps(dataset))
store_file.close()
