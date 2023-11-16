'''
This script can be used to make figure 5, assuming you made the required
data from the corresponding test script
'''

import json
import matplotlib.pyplot as plt
import numpy as np

def parse(d_json):
    keys = ['(64, 1)', '(128, 1)', '(256, 1)', '(512, 1)']
    return[np.mean(d_json['results'][k]) for k in keys]

# output files from gaussian_exact_d_effect with different d settings
d5_filename = "YOUR_FILE_NAME" 
d10_filename = "YOUR_FILE_NAME"
d25_filename = "YOUR_FILE_NAME"
d50_filename = "YOUR_FILE_NAME"

d5_file   = open(d5_filename, "r")
d10_file  = open(d10_filename, "r")
d25_file  = open(d25_filename, "r")
d50_file  = open(d50_filename, "r")

d5_json   = json.load(d5_file)
d10_json  = json.load(d10_file)
d25_json  = json.load(d25_file)
d50_json  = json.load(d50_file)

d5_file.close()
d10_file.close()
d25_file.close()
d50_file.close()

d = [64,128,256,512]
r5 = parse(d5_json)
r10 = parse(d10_json)
r25 = parse(d25_json)
r50 = parse(d50_json)

plt.plot(np.log(d), np.log(r5))
plt.plot(np.log(d), np.log(r10))
plt.plot(np.log(d), np.log(r25))
plt.plot(np.log(d), np.log(r50))
plt.plot([np.log(64), np.log(512)],[-3.5, -3.5 - np.log(512) + np.log(64)], '--')
plt.legend(['d=5','d=10','d=25','d=50','slope=-1'])
plt.xlabel('Log(n)',fontsize=14)
plt.ylabel('Log of Squared Error',fontsize=14)
plt.title(r'Effect of $d$ (Gaussian, $k=1,\varepsilon=2.0$)',fontsize=15)
plt.tight_layout()
plt.show()
#plt.savefig("YOUR_FILE_NAME.pdf") # if you want to save the figure
