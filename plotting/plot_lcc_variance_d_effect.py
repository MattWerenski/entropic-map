'''
This script can be used to make figure 6, assuming you made the required
data from the corresponding test script
'''

import json
import matplotlib.pyplot as plt
import numpy as np

def parse(d_json):
    keys = ['(64, 1)', '(128, 1)', '(256, 1)', '(512, 1)']
    return[np.mean(d_json['replace'][k]) for k in keys]

# output files from lcc_variance_d_effect
d5_name   = "YOUR_FILE_NAME"
d10_name  = "YOUR_FILE_NAME"
d25_name  = "YOUR_FILE_NAME"

d5_file   = open(d5_name, "r")
d10_file  = open(d10_name, "r")
d25_file  = open(d25_name, "r")

d5_json   = json.load(d5_file)
d10_json  = json.load(d10_file)
d25_json  = json.load(d25_file)

d5_file.close()
d10_file.close()
d25_file.close()

d = [64,128,256,512]
r5 = parse(d5_json)
r10 = parse(d10_json)
r25 = parse(d25_json)

plt.plot(np.log(d), np.log(r5))
plt.plot(np.log(d), np.log(r10))
plt.plot(np.log(d), np.log(r25))

plt.plot([np.log(64), np.log(512)],[0, 0 - np.log(512) + np.log(64)], '--')

plt.legend(['d=5','d=10','d=25','slope=-1'])
plt.xlabel('Log(m)',fontsize=14)
plt.ylabel('Log(Variance)',fontsize=14)
plt.title(r'Effect of $d$ (Log-Concave, $\varepsilon=5.0$)',fontsize=15)
plt.tight_layout()
plt.show()
#plt.savefig('YOUR_FILE_NAME.pdf')