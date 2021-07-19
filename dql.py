from collections import deque
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
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
import matplotlib
matplotlib.use( 'tkagg' )
from matplotlib import style

style.use("ggplot")
# config = tf.ConfigProto( device_count = {'GPU': 1} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

#Global Variables
NO_EPISODES = 1000
TRAIN_END = 0
current_state = ''
current_action = 0
is_first = True
invalid_action = False
latencies = []
total_latency = []
rewards = []
total_rewards = []
mean_latency = 0
beta = 7
batch_size = 192
REQUIRED_LATENCY = 4.00009631


def calculate_required_latency(message_size,bandwidth_of_handle):
    pass
#Hyper Parameters
def discount_rate(): #Gamma
    return 0.95

def learning_rate(): #Alpha
    return 0.001


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

def get_state_input(state):

    global beta
    output = []
    
    no_of_edge_devices = len(state["PR"])
         
    for a in range(beta):

        if a < no_of_edge_devices:
            output.append(state['PR'][a])
        else:
            output.append(15)
    
    for a in range(beta):

        if a < no_of_edge_devices:
            output.append(state['memories'][a][0])
            output.append(state['memories'][a][1])
        else:
            output.append(0)
            output.append(0)
    
    for a in range(beta):

        if a < no_of_edge_devices:
            output.append(state['bandwidth'][a][0])
            output.append(state['bandwidth'][a][1])
        else:
            output.append(-1)
            output.append(-1)
    
    output.append(state['input_size'])
    return output




class DeepQNetwork():
    def __init__(self, states, actions, alpha, reward_gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.reward_gamma = reward_gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []

                
    def build_model(self):

        model = keras.Sequential() 
        model.add(keras.layers.Dense(self.nS*2, input_dim=self.nS, activation='relu')) 
        
        # model.add(keras.layers.Dense(self.ns*4, activation='relu')) 

        model.add(keras.layers.Dense(self.nA, activation='softmax')) 

        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) 
        return model
    
    def get_action(self,state):
        global is_first
        global current_state
        global current_action
        global gamma
        global epsilon
        global EPS_DECAY
        global beta
        global invalid_action

        # print(state)

        no_of_edges = len(state['PR'])

        # print(state)
        subsets = get_subsets(set([x for x in range(beta)]))

        # if new_state['bandwidth'] not in all_bands:
        #     all_bands.append(new_state['bandwidth'])
        
        random_value = np.random.uniform(0,1)

        valid_subsets = get_subsets(set([x for x in range(no_of_edges)]))

        state_flattened = get_state_input(state)

        state_flattened = np.array(state_flattened).reshape(1,self.nS)

        if np.random.rand() <= self.epsilon:

            action = np.random.randint(0,self.nA)
        else:
            action_vals = self.model.predict(state_flattened) #Exploit: Use the NN to predict the correct action from this state
        
            action = np.argmax(action_vals[0])
        
        current_state = state_flattened
        current_action = action
        
        return_arr = subsets[action]

        valid = False
        possible_valid_actions = list()

        for valid_subset in valid_subsets:
            if return_arr == valid_subset:
                valid = True
                break

            elif len(list(set(return_arr) & set(valid_subset))) > 0:
                possible_valid_actions = list(set(return_arr) & set(valid_subset))
    
        is_first = True
        gamma = len(return_arr)
        
        if valid:
            invalid_action = False
            return return_arr
        elif len(possible_valid_actions) > 0:
            invalid_action = False
            return possible_valid_actions
        else:
            invalid_action = True
            return valid_subsets[0]


    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward):
        #Store the experience in memory
        self.memory.append( (state, action, reward) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        # print("Training started")
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        # nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            # nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        # nst_predict = self.model.predict(nst)
        index = 0
        for state, action, reward in minibatch:
            x.append(state)
            #Predict from state
            # nst_action_predict_model = nst_predict[index]
            # if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
            #     target = reward
            # else:   #Non terminal
            #     target = reward + self.gamma * np.amax(nst_action_predict_model)
            target = reward
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1

        
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 60 #Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        loss_sum = 0
        for i in range(epoch_count):
            loss_sum += hist.history['loss'][i]
        self.loss.append(loss_sum / epoch_count)
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def reward(self, obs_latency):
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
        global batch_size
        global invalid_action

        # print("came in reward")
        if is_first:
            # print("came in is first")
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

            if invalid_action:
                reward -= 1000
            
            # print("reward calculated = ",reward)
            self.store(current_state, current_action, reward)
            

            rewards.append(reward)
                
            if len(self.memory) > batch_size:
                self.experience_replay(batch_size)

        else:
            pass
            # print("Didnt run")  


#Create the agent
nS = 5*beta + 1
nA = 2**beta - 1
dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.0001)
print(dqn.model.summary())


for episode in range(NO_EPISODES):

    episode_reward = 0

    driver(dqn.get_action,dqn.reward)

    # print("latencies : ", latencies)

    # print("rewards : ", rewards)

    avg_reward = sum(rewards) / len(rewards)

    print("episode: {}/{}, score: {}, average latency: {}, e: {},"
                  .format(episode+1, NO_EPISODES, sum(rewards), np.mean(latencies),dqn.epsilon))
    
    if avg_reward > -250: 
        total_rewards.append(avg_reward)
        total_latency.append(sum(latencies)/len(latencies))
    
    latencies = []
    rewards = []



temp_rewards = [np.mean(total_rewards[i:i+50]) for i in range(0,len(total_rewards),50)]
temp_latencies = [np.mean(total_latency[i:i+50]) for i in range(0,len(total_latency),50)]
temp_loss = [np.mean(dqn.loss[i:i+50]) for i in range(0,len(dqn.loss),50)]

f = plt.figure(1)

plt.plot(np.arange(len(temp_rewards)), temp_rewards)

f.show()

f.savefig("dql_episodic_rewards.png")


g = plt.figure(2)

plt.plot(np.arange(len(temp_latencies)), temp_latencies)

g.show()

g.savefig("dql_episodic_latency.png")

h = plt.figure(3)

plt.plot(np.arange(len(temp_loss)), temp_loss)

h.show()

h.savefig("dql_loss.png")




