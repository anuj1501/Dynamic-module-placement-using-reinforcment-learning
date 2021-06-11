import numpy as np  
from PIL import Image  
import pickle  
import time  
from main import driver 
import math
import json
import sys
import matplotlib.pyplot as plt

NO_EPISODES = 1 
epsilon = 1
EPS_DECAY = 0.01 

start_q_table = None  

LEARNING_RATE = 0.1
DISCOUNT = 0.95


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

mean_latency = 0

gamma = 0

beta = 3

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

def get_action(state):
    global is_first
    global q_table
    global current_state
    global current_action
    global gamma
    global epsilon
    global EPS_DECAY

    subsets = get_subsets(set([0,1,2]))

    new_state = {}

    new_state['PR'] = state['PR']
    new_state['input_size'] = math.floor(state['input_size'] - state['input_size']%10)

    new_state['memories'] = {}
    new_state['bandwidth'] = {}
    for a in range(0,3):
        new_state['memories'][a] = (math.floor(state['memories'][a][0] - state['memories'][a][0]%200),math.floor(state['memories'][a][1] - state['memories'][a][1]%200))
        new_state['bandwidth'][a] = (math.floor(state['bandwidth'][a][0] - state['bandwidth'][a][0]%6),math.floor(state['bandwidth'][a][0] - state['bandwidth'][a][0]%6))

    new_state = json.dumps(new_state)
    if q_table.get(new_state) is None:
        q_table[new_state] =  [np.random.uniform(-5, 0) for a in range(7)]
   
    q_vals = q_table[new_state]

    random_value = np.random.uniform(0,1)

    if random_value > epsilon:

        action = np.argmax(q_vals)

    else:

        action = np.random.randint(0,len(q_vals))

    print("action :",action)
    epsilon -= EPS_DECAY

    current_state = new_state
    current_action = action


    print("action : ",action)
    print(subsets)
    return_arr = subsets[action]
    is_first = True
    print(return_arr)
    print("List of edge indices : ",return_arr)

    gamma = len(return_arr)

    return return_arr
    

def reward(obs_latency):
    #Store previous reward and update 
    #reward = curr_reward + gamma*(prev_reward)
    global is_first
    global current_state
    global current_action
    global latencies
    global mean_latency
    global beta
    global gamma
    # print("visited")
    if is_first:
        # print("Reward function {}".format(obs_latency))

        latencies.append(obs_latency)

        median_latency = np.median(latencies)

        lamda = obs_latency - median_latency

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

        q_table[current_state][current_action] += reward 

    else:
        print("Didnt run")  



# sys.stdout = open("test.txt", "w")

for episode in range(NO_EPISODES):

    episode_reward = 0

    print('EPISODE NUMBER {}'.format(episode))

    time.sleep(5)

    driver(get_action,reward)

    epsilon *= EPS_DECAY

print(latencies)
plt.plot(np.arange(len(latencies)), latencies)
plt.show()
# sys.stdout.close()

# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
