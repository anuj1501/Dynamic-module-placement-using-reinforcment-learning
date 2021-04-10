import numpy as np  
from PIL import Image  
import pickle  
import time  
from main import driver 



NO_EPISODES = 1
MOVE_PENALTY = 1  
ENEMY_PENALTY = 300  
FOOD_REWARD = 25  
epsilon = 0.5 
EPS_DECAY = 0.9999 

start_q_table = None  

LEARNING_RATE = 0.1
DISCOUNT = 0.95







if start_q_table is None:
    q_table = {}

    for band in range(1, 3):
        for comp in range(1,5):
            for peak_memory in range( 1, 5):
                for exec_time in range( 1, 10):
                    q_table[ (band,comp,peak_memory,exec_time) ] = [np.random.uniform(-5, 0) for a in range(8)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)



def get_action(state):

    print("GETTING ACTION")
    

# def update_q(state,action,obs_latency):


for episode in range(NO_EPISODES):

    episode_reward = 0

    print('EPISODE NUMBER {}'.format(episode))

    time.sleep(5)

    driver(get_action)




    epsilon *= EPS_DECAY



# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
