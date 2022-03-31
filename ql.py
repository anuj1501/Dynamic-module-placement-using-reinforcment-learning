import numpy as np  
import pandas as pd
from PIL import Image  
import pickle  
import time  
from main import driver 
import math
import json
import sys
import collections
import matplotlib.pyplot as plt
import itertools 

total_rewards = []
total_latency = []
NO_EPISODES = 1000
epsilon = 0.4
min_epsilon = 0.1
EPS_DECAY = 0.01
reward_gamma = 0.5 
first_action = False
start_q_table = None  
previous_state = None
previous_action = None
valid_action = False
LEARNING_RATE = 0.1
DISCOUNT = 0.95
REQUIRED_LATENCY = 1.5 #3.00009631


# #Ranges :-
# Bandwidth :- 0-26 (6)
# Message inst :- 95000 - 168000 (1000)
# Peak-memories :- 200-600 (200)
# input-size :- 900-1100 (10)
# {
# {
#     'PR': {0: 8, 1: 3, 2: 5}, 
#     'memories': {0: (600, 19400), 1: (400, 19600), 2: (400, 19600)}, 
#     'bandwidth': {0: (6.275, 6.275), 1: (6.5, 6.5), 2: (5.325, 5.325)}, 
#     'input_size': 1000
# } : [.....]
# }

latencies = []
rewards = []
mean_latency = 0

gamma = 0

beta = 4

if start_q_table is None:
    q_table = {}
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

#     for band in range(0, 31, 5):
#         for comp in range(1,5):
#             for peak_memory in range( 1, 5):
#                 for exec_time in range( 1, 10):
#                     q_table[ (band,comp,peak_memory,exec_time) ] = [np.random.uniform(-5, 0) for a in range(8)]



# permutations = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]


def get_subsets(fullset):
  listrep = list(fullset)
  subsets = []
  for i in range(2**len(listrep)):
    subset = []
    for k in range(len(listrep)):
      if i & 1<<k:
        subset.append(listrep[k])
    subsets.append(subset)
  return subsets[1:]

current_state = ''
current_action = 0
is_first = True


# def get_first():
#     global is_first
#     return is_first

# def first_false():
#     global is_first
#     is_first = False

# def first_true():
#     global is_first
#     is_first = True
all_bands = []

def get_action(state, required_lat):
    global is_first
    global q_table
    global current_state
    global current_action
    global gamma
    global epsilon
    global EPS_DECAY
    global beta

    # print(state)

    no_of_edges = len(state['PR'])

    # print(state)
    subsets = get_subsets(set([x for x in range(beta)]))

    new_state = {}

    new_state['PR'] = state['PR']
    new_state['input_size'] = math.floor(state['input_size'] - state['input_size']%100)

    new_state['memories'] = {}
    new_state['bandwidth'] = {}

    for a in range(0,len(state['PR'])):
        new_state['memories'][a] = (math.floor(state['memories'][a][0] - state['memories'][a][0]%200),math.floor(state['memories'][a][1] - state['memories'][a][1]%200))
        new_state['bandwidth'][a] = (math.floor(state['bandwidth'][a][0] - state['bandwidth'][a][0]%12),math.floor(state['bandwidth'][a][0] - state['bandwidth'][a][0]%12))
    
    for a in range(len(state['PR']), beta):
        new_state['PR'][a] = 15
        new_state['bandwidth'][a] = (-1,-1)
        new_state['memories'][a] = (0,0)

    if new_state['bandwidth'] not in all_bands:
        all_bands.append(new_state['bandwidth'])

    new_state = json.dumps(new_state)
    # print new_state
    # print "\n"*2
    if q_table.get(new_state) is None:
        q_table[new_state] =  [np.random.uniform(-5, 0) for a in range(2**beta - 1)]
   
    q_vals = q_table[new_state]
    random_value = np.random.uniform(0,1)

    # print(state)
    # print(no_of_edges)
    valid_subsets = get_subsets(set([x for x in range(no_of_edges)]))

    if random_value > epsilon:

        action = np.argmax(q_vals)

    else:

        action = np.random.randint(0,len(q_vals))

    # {"PR" : {0: 4 ,1:5 , 2:6 , {}}}
    # {"PR" : {0: 5 ,1:4 , 2:6 }
    # print("action :",action)
    current_state = new_state
    current_action = action


    # print("action : ",action)
    # print(subsets)
    
    return_arr = subsets[action]
    # print(subsets)
    # print(valid_subsets)
    # print(return_arr)

    valid = False

    for valid_subset in valid_subsets:
        # print("lol")
        # print(type(return_arr))
        # print(type(valid_subset))
        if return_arr == valid_subset:

            valid = True

            break
    

    is_first = True
    # print(return_arr)
    # print("List of edge indices : ",return_arr)

    gamma = len(return_arr)

    if valid:
        # print("action : ", return_arr)
        return return_arr
        valid_action = True
    else:
        # print("scam")
        q_table[new_state][action] -= 10000
        valid_action = False
        return valid_subsets[0]
    

def reward(obs_latency):
    # print("latency : ", obs_latency)
    #Store previous reward and update 
    #reward = curr_reward + gamma*(prev_reward)
    global is_first
    global current_state
    global current_action
    global latencies
    global REQUIRED_LATENCY
    global beta
    global gamma
    global rewards
    global first_action
    global previous_state
    global reward_gamma
    if is_first:
        # print("Reward function {}".format(obs_latency))
        # print obs_latency
        latencies.append(obs_latency)

        # median_latency = np.median(latencies)

        lamda = obs_latency - REQUIRED_LATENCY

        reward = 0

        delta = 0

        alpha = 0.001

        if lamda < 0:

            delta = (alpha * np.exp(-1 * lamda))
        else:

            delta = (alpha * np.exp(lamda))
        
        if lamda == 0:

            reward = 0

        elif lamda > 0 and beta - gamma == 1:

            reward = 0

        elif lamda > 0 and beta - gamma > 1:

            reward = (-1 * np.exp(beta - gamma - 1) * delta)

        elif lamda > 0:
            
            reward = (-1 * np.exp(gamma) * delta)

        is_first = False

        if first_action == False:
            first_action = True
        else:
            if valid_action:
                q_table[previous_state][previous_action] += (reward_gamma * reward)
            q_table[current_state][current_action] += reward 
        
        previous_state = current_state
        previous_action = current_action
        rewards.append(reward)

    else:
        pass
        # print("Didnt run")  



# sys.stdout = open("test.txt", "w")

for episode in range(NO_EPISODES):

    episode_reward = 0

    print('EPISODE NUMBER {}'.format(episode))

    # time.sleep(5)

    driver(get_action,reward)
    # print("latencies : ", latencies)
    total_latency.append(np.mean(latencies))
    latencies= []
    total_rewards.append(sum(rewards))
    rewards = []
    if epsilon > min_epsilon:
        epsilon -= EPS_DECAY
    # print("length of bandwidth space is : ", len(all_bands))
    # print("state size : ", len(q_table))


# for item in all_bands:
#     print(item)
# print("epsilon : ", epsilon)
# print(q_table)
step_latency = []

for i in range(0,NO_EPISODES,10):

    temp = total_latency[i:i+10]

    avg = (sum(temp) / len(temp))
    step_latency.append(avg)

out = dict(itertools.islice(q_table.items(), 10))
print(out)
plt.plot(np.arange(len(step_latency)), step_latency)
# print(total_latency)
plt.show()
plt.savefig("episodic_latency.png")


# sys.stdout.close()

# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
