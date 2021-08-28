#                     if message.name == "M.A":
#                         module = "ServiceA"
                    
#                     else:
#                         module = "Actuator"

#                     alloc_des_reverse = {v: k for k, v in alloc_DES.iteritems()}

#                     if module in self.apps[message.app_name].get_sink_modules():
#                         """
#                         The module is a SINK (Actuactor)
#                         """
#                         id_node = self.alloc_DES[des]
#                         time_service = 0
#                     else:
#                         """
#                         The module is a processing module
#                         """
#                         id_node = self.alloc_DES[des]

#                         # att_node = self.topology.get_nodes_att()[id_node] # WARNING DEPRECATED from V1.0
#                         att_node = self.topology.G.nodes[id_node]

#                         time_service = message.inst / float(att_node["IPT"])
                    
# self.metrics_for_devices.append(
# {"id": message.id, "type": self.LINK_METRIC, "app": message.app_name, "module": module, "message": message.name,
# "DES.src": self.alloc_des_reverse[link[0]], "DES.dst": self.alloc_des_reverse[link[1]], "module.src": message.src,
# "TOPO.src": link[0], "TOPO.dst": link[1],

# "service": time_service, "time_in": self.env.now + self.add_time,
# "time_out": time_service + self.env.now + self.add_time, "time_emit": float(message.timestamp) + self.add_time,
# "time_reception": float(message.timestamp_rec) + self.add_time

# })

# def get_subsets(fullset):
#   listrep = list(fullset)
#   subsets = []
#   for i in range(2**len(listrep)):
#     subset = []
#     for k in range(len(listrep)):            
#       if i & 1<<k:
#         subset.append(listrep[k])
#     subsets.append(subset)        
#   return subsets
# subsets = get_subsets(set([1,2,3,4]))
# subsets = subsets[1:]
# subsets.sort(key=lambda x: len(x))
# # subsets.sort(key=lambda x: x[0])
# print(subsets)
# print(len(subsets))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import style

style.use("ggplot")

data = pd.read_csv("Obtained Results.csv")


latency_deviation = data["latency_deviation"].values

temp_deviation = [np.mean(latency_deviation[i:i+10]) for i in range(0,len(latency_deviation),10)]

plt.plot(np.arange(len(temp_deviation)), temp_deviation)

plt.savefig("latency deviation.png")



rewards = data["total_rewards"].values

temp_rewards = [np.mean(rewards[i:i+10]) for i in range(0,len(rewards),10)]

plt.plot(np.arange(len(temp_rewards)), temp_rewards)

plt.savefig("dql_episodic_rewards.png")