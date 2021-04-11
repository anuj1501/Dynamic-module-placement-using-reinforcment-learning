import numpy as np  
from PIL import Image  
import pickle  
import time  
from main import driver 
import math
import json

NO_EPISODES = 1
MOVE_PENALTY = 1  
ENEMY_PENALTY = 300  
FOOD_REWARD = 25  
epsilon = 0.5 
EPS_DECAY = 0.9999 

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


subsets = get_subsets(set([0,1,2]))


def get_action(state):

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
    action = np.argmax(q_vals)

    return_arr = subsets[action]

    return return_arr
    

# def update_q(state,action,obs_latency):


for episode in range(NO_EPISODES):

    episode_reward = 0

    print('EPISODE NUMBER {}'.format(episode))

    time.sleep(5)

    driver(get_action)




    epsilon *= EPS_DECAY



# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
