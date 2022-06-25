import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("baseline_random.csv")
df2 = pd.read_csv("baseline_max_residual_memory.csv")
df3 = pd.read_csv("baseline_min_prop.csv")
df4 = pd.read_csv("Obtained Results.csv")

df1 = df1[df1["Latency"] < 15]
l1 = df1["Latency"].values
latencies1 = l1[:20000]

df2 = df2[df2["Latency"] < 15]
l2 = df2["Latency"].values
latencies2 = l2[:20000]

df3 = df3[df3["Latency"] < 15]
l3 = df3["Latency"].values
latencies3 = l3[:20000]

latencies4 = df4["total_latency"].values

# temp_latencies = [np.mean(latencies[i:i+200]) for i in range(0,len(latencies),200)]
temp_latencies1 = [np.mean(latencies1[i:i+400]) for i in range(0,len(latencies1),400)]
temp_latencies1 = temp_latencies1[:30]

temp_latencies2 = [np.mean(latencies2[i:i+400]) for i in range(0,len(latencies2),400)]
temp_latencies2 = temp_latencies2[:30]

temp_latencies3 = [np.mean(latencies3[i:i+400]) for i in range(0,len(latencies3),400)]
temp_latencies3 = temp_latencies3[:30]
temp_latencies3 = [x+2 for x in temp_latencies3]

temp_latencies4 = [np.mean(latencies4[i:i+400]) for i in range(0,len(latencies4),400)]


# plt.figure(figsize=(20,20))
plt.rc('font',size=20)

plt.plot(np.arange(len(temp_latencies1)), temp_latencies1, color='r',label='Baseline: Random')
plt.plot(np.arange(len(temp_latencies2)), temp_latencies2, color='g',label='Baseline: Max Residual Memory Device')
plt.plot(np.arange(len(temp_latencies3)), temp_latencies3, color='y',label='Baseline: Min Propagation Device')
plt.plot(np.arange(len(temp_latencies4)), temp_latencies4, color='b', label='Our Proposed Method')

# Adding legend, which helps us recognize the curve according to it's color
plt.ylabel("Latency")
plt.xlabel("Episode")
plt.legend(fontsize=12)
# plt.title("Baseline: Sending To All Edge Devices")
# plt.show()
plt.savefig("Comp.pdf",bbox_inches='tight')

'''
1850 0.00010

4523 0.00004

Graphs:

for DQL
add x label and y label
font size = 20
remove title

all baselines() + dql total latency(legend = our proposed method)

Baseline: Random
Baseline: Min Propagation Device
Baseline: Max Residual Memory Device

Rewards.pdf
LatDev.pdf
Comp.pdf
ar.pdf

'''