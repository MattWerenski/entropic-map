'''
This script can be used to make figure 3, assuming you made the required
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

# files output by gmm_variance_eps_effect
eps1_name  = "YOUR_FILE_NAME"
eps2_name  = "YOUR_FILE_NAME"
eps5_name  = "YOUR_FILE_NAME" 
eps10_name = "YOUR_FILE_NAME"

eps1_file   = open(eps1_name, "r")
eps2_file   = open(eps2_name, "r")
eps5_file   = open(eps5_name, "r")
eps10_file  = open(eps10_name, "r")

eps1_json  = json.load(eps1_file)
eps2_json  = json.load(eps2_file)
eps5_json  = json.load(eps5_file)
eps10_json = json.load(eps10_file)

eps1_file.close()
eps2_file.close()
eps5_file.close()
eps10_file.close()

res1 = make_dataset(eps1_json)
res2 = make_dataset(eps2_json)
res5 = make_dataset(eps5_json)
res10 = make_dataset(eps10_json)


mrange = [64,128,192,256,320,384,448,512]
data1 = [res1[m][1].mean() for m in mrange]
data2 = [res2[m][1].mean() for m in mrange]
data5 = [res5[m][1].mean() for m in mrange]
data10 = [res10[m][1].mean() for m in mrange]

plt.plot(np.log(mrange),np.log(data1))
plt.plot(np.log(mrange),np.log(data2))
plt.plot(np.log(mrange),np.log(data5))
plt.plot(np.log(mrange),np.log(data10))
plt.plot([np.log(64), np.log(512)], [-1.5, -1.5 - np.log(512) + np.log(64)], '--')

plt.legend([r"$\varepsilon$=1.0",r"$\varepsilon$=2.0",r"$\varepsilon$=5.0",r"$\varepsilon$=10.0","Slope = -1"])
plt.xlabel(r"$\log(m)$",fontsize=14)
plt.ylabel(r"$\log(Variance)$",fontsize=14)
plt.title(r"Impact of $m$ and $\varepsilon$ on the Variance (5D GMM)",fontsize=15)
plt.tight_layout()
plt.savefig("figures/variance_2/log_concave/m_eps_replace.pdf")