import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("Obtained Results.csv")

latencies = df["latency_deviation"].values

temp_latencies = [np.mean(latencies[i:i+100]) for i in range(0,len(latencies),100)]

plt.plot(np.arange(len(temp_latencies)), temp_latencies)
plt.xlabel("Epsilon values: 1 equivalent to average of 200")
plt.ylabel("Access Rate: 1 equivalent to average of 200")
plt.show()

'''
1850 0.00010

4523 0.00004


'''