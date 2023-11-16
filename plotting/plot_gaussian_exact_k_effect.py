'''
This script can be used to make figures 1 and 4, assuming you made the required
data from the corresponding test script
'''

import json
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

def make_dataset(json_structure):
    # helper function for parsing json files
    j_data = json_structure["results"]
    
    results = {}
    
    for key in j_data:
        (n, k) = literal_eval(key)
        r = np.array(j_data[key])
        
        if not n in results:
            results[n] = {}
        
        results[n][k] = r
    
    return results

filename = "YOUR_FILE_NAME" # output file from gaussian_exact_k_effect
json_file   = open(filename, "r")
json_structure = json.load(json_file)
json_file.close()
results = make_dataset(json_structure)

ns = []
ks = {}
for n in results:
    ns += [n]
    for k in results[n]:
        if not k in ks:
            ks[k] = []
        ks[k] += [np.mean(results[n][k])]
        
for k in ks:
    plt.plot(np.log(ns),np.log(ks[k]))

plt.plot([np.log(64),np.log(512)],[-0.8, -0.8 - np.log(512) + np.log(64)],'--')
    
plt.title(r"Error in Estimate of $T_\varepsilon$ (5D Gaussians, $\varepsilon = 10.0$)",fontsize=15)
plt.legend([r"$k=1$",r"$k=2$",r"$k=4$",r"$k=8$",r"$k=16$",r"$k=32$",r"slope=-1"])    

plt.xlabel(r"$\log(n)$", fontsize=14)
plt.ylabel(r"log of Squared Error", fontsize=14)
plt.tight_layout()
plt.show()
#plt.savefig("YOUR_FILE_NAME.pdf") # if you want to save the figure