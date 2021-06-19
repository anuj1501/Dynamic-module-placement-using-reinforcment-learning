import pandas as pd
import numpy as np

df = pd.read_csv('output.csv')
ipt_array = []
for num in range(6, len(df["cycles"])):
    #ipt_val = float( df["cycles"][num] ) / float( df["exe_time"][num] )
    # ipt_array.append(ipt_val)
    ipt_array.append(df["cycles"][num]/(26.8*8))
print(np.mean(ipt_array))