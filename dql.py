'''

changed action to self.state to action mapper(state_str)


'''
from collections import deque
# from typing_extensions import final
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import argparse
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
import os
style.use("ggplot")
# config = tf.ConfigProto( device_count = {'GPU': 1} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

#Global Variables
NO_EPISODES = 0
TRAIN_END = 0
current_state = ''
current_action = 0
is_first = True
invalid_action = False
latencies = []
total_latency = []
rewards = []
total_rewards = []
epsilon_values = []
deviations = []
latency_deviation = []
access_rate = []
episode_access_rate = []
avg_waiting_time = [0]
exploit_or_explore = []
total_exploit_or_explore = []
mean_latency = 0
beta = 10
batch_size = 512
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

    '''
    {'PR': {0: 7, 1: 9, 2: 3, 3: 10, 5: 9, 6: 2, 7: 7, 8: 5, 9: 10}, 
    'memories': {0: (600, 19400), 1: (600, 19400), 2: (600, 19400), 3: (200, 19800), 5: (400, 19600), 
    6: (400, 19600), 7: (200, 19800), 8: (400, 19600), 9: (200, 19800)}, 
    'bandwidth': {0: (3.357142857142857, 3.357142857142857), 1: (2.7285714285714286, 2.7285714285714286), 
    2: (3.7142857142857144, 3.7142857142857144), 3: (2.9714285714285715, 2.9714285714285715), 
    5: (3.6285714285714286, 3.6285714285714286), 6: (3.6142857142857143, 3.6142857142857143), 
    7: (3.5714285714285716, 3.5714285714285716), 8: (3.6714285714285713, 3.6714285714285713), 
    9: (3.6, 3.6)}, 'input_size': 1100}
    '''

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
        self.map_of_state_best_reward = dict()
        self.map_of_state_best_action = dict()
        self.alpha = alpha
        self.reward_gamma = reward_gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        if os.path.exists("DQN_Model.h5"):
            print("------------------FOUND EXISTING MODEL----------------------------")
            self.model.load_weights("DQN_Model.h5")
        self.loss = []
        self.val_loss = []

                
    def build_model(self):

        model = keras.Sequential() 
        model.add(keras.layers.Dense(self.nS*2, input_dim=self.nS, activation='sigmoid')) 
        
        model.add(keras.layers.Dense(self.nS*4, activation='sigmoid')) 

        model.add(keras.layers.Dense(self.nA, activation='softmax')) 

        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) 
        return model
    
    def get_action(self,state,required_lat):
        global is_first
        global current_state
        global current_action
        global gamma
        global epsilon
        global EPS_DECAY
        global beta
        global invalid_action
        global episode_access_rate
        global REQUIRED_LATENCY
        global avg_waiting_time
        global exploit_or_explore

        REQUIRED_LATENCY = required_lat
        # print(required_lat)
        # print(state)

        # print("check 1")
        no_of_edges = len(state['PR'])

        # print(state)
        subsets = get_subsets(set([x for x in range(beta)]))
        # print("check 2")
        # if new_state['bandwidth'] not in all_bands:
        #     all_bands.append(new_state['bandwidth'])
        
        random_value = np.random.uniform(0,1)

        valid_subsets = get_subsets(set([x for x in range(no_of_edges)]))
        # print("check 3")
        state_flattened = get_state_input(state)

        state_flattened = np.array(state_flattened).reshape(1,self.nS)
        # print("check 4")
        if np.random.rand() <= self.epsilon:
            exploit_or_explore.append("explore")
            action = np.random.randint(0,self.nA)
        else:
            exploit_or_explore.append("exploit")
            action_vals = self.model.predict(state_flattened) #Exploit: Use the NN to predict the correct action from this state
        
            action = np.argmax(action_vals[0])
        # print("check 5")
        current_state = state_flattened
        current_action = action
        
        return_arr = subsets[action]
        # print("predicted_action = ",return_arr)
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
            # print("lol")
            invalid_action = False
            episode_access_rate.append(float(len(return_arr))/no_of_edges)
            return return_arr
        elif len(possible_valid_actions) > 0:
            invalid_action = False
            episode_access_rate.append(float(len(possible_valid_actions))/no_of_edges)
            return possible_valid_actions
        else:
            invalid_action = True
            episode_access_rate.append(float(len(valid_subsets[0]))/no_of_edges)
            return valid_subsets[0]

    def create_target_vector_with_lambda(self, tg_vector,reward,action):
        lambda_reward = 0.1
        
        exponential_reward = math.exp(reward*lambda_reward)

        target_f_modified = np.zeros(tg_vector.shape)
        
        temp = (1 - exponential_reward)/(np.sum(tg_vector) - tg_vector[action])
        
        for i in range(target_f_modified.shape[0]):

            if i==action:
                target_f_modified[i] = exponential_reward
            else:
                target_f_modified[i] = (temp * tg_vector[i])

        return target_f_modified 
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
            state_str = np.array_str(state)
            if state_str not in self.map_of_state_best_action.keys():
                self.map_of_state_best_action[state_str] = action
                self.map_of_state_best_reward[state_str] = reward
            
            else:
                if self.map_of_state_best_reward[state_str] < reward:
                    self.map_of_state_best_action[state_str] = action
                    self.map_of_state_best_reward[state_str] = reward

            #Predict from state
            # nst_action_predict_model = nst_predict[index]
            # if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
            #     target = reward
            # else:   #Non terminal
            #     target = reward + self.gamma * np.amax(nst_action_predict_model)
            # target = reward
            # target_f = st_predict[index]
            
            # target_f[action] = target
            # target_f[self.map_of_state_best_action[state_str]] = 1
            target_f_with_lambda = self.create_target_vector_with_lambda(st_predict[index],reward, action)
            y.append(target_f_with_lambda)
            index += 1

        
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 30 #Epochs is the number or iterations
        t4 = time.time()  
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, validation_split = 0.2, verbose=0)
        # print("time taken to train the model: ",time.time() - t4)
        #Graph Losses
        loss_sum = 0
        val_loss_sum = 0
        for i in range(epoch_count):
            loss_sum += hist.history['loss'][i]
            val_loss_sum += hist.history['val_loss'][i]

        self.loss.append(loss_sum / epoch_count)
        self.val_loss.append(val_loss_sum / epoch_count)
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
        global deviations
        global avg_waiting_time

        
        # med_latency = (REQUIRED_LATENCY + (np.mean(avg_waiting_time)))
        med_latency = REQUIRED_LATENCY
        # print("required latency: ",REQUIRED_LATENCY)
        avg_waiting_time.append(abs(obs_latency - REQUIRED_LATENCY))
        # print("obs_latency: ",obs_latency)
        # print("REQUIRED_LATENCY: ", REQUIRED_LATENCY)
        # print()
        # print("came in reward")
        if is_first and abs(obs_latency - REQUIRED_LATENCY) < 20:
            # print "required latency : ", REQUIRED_LATENCY
            # print "actual latency : ",obs_latency
            # print("came in is first")
            # print("Reward function {}".format(obs_latency))
            # print("the actual latency = ",obs_latency)
            latencies.append(obs_latency)
            # print("required_latency = ",REQUIRED_LATENCY)
            deviations.append(obs_latency - med_latency)

            # print("check 1")
            # median_latency = np.median(latencies)

            lamda = (obs_latency - med_latency)

            reward = 0

            delta = 0

            alpha = 0.001

            if lamda < 0:

                delta = (alpha * np.exp(-1 * lamda))
            else:

                delta = (alpha * np.exp(lamda))
            
            if lamda == 0:

                reward = 0

            elif lamda > 0 and beta - gamma == 0:

                reward = 0

            elif lamda > 0 and beta - gamma > 0:

                reward = (-1 * np.exp(beta - gamma - 1) * delta)

            elif lamda > 0:
                
                reward = (-1 * np.exp(gamma) * delta)

            is_first = False

            if invalid_action:
                reward -= 1000
            
            # print("reward calculated = ",reward)
            self.store(current_state, current_action, reward)
            # print("check 1")
            # print(reward)

            if reward > -300:

                rewards.append(reward)
                
            if len(self.memory) > batch_size:
                t1 = time.time() 
                self.experience_replay(batch_size)
                # print("time to run experience replay: ",time.time() - t1)

        else:
            pass
            # print("Didnt run")  

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--Episodes", help = "Show Output")
 
# Read arguments from command line
args = parser.parse_args()
# print(args)
if args.Episodes:
    NO_EPISODES = int(args.Episodes)
else:
    NO_EPISODES = 100


nS = 5*beta + 1
nA = 2**beta - 1
dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.00004)
print(dqn.model.summary())


for episode in range(NO_EPISODES):
    t3 = time.time() 
    episode_reward = 0

    t2 = time.time()
    driver(dqn.get_action,dqn.reward)
    # print("time taken for the simulation is: ",time.time() - t2)

    # print("latencies : ", latencies)

    # print("rewards : ", rewards)

    if episode > 6000:
        dqn.epsilon = 0

    if len(rewards) > 0:
        avg_reward = sum(rewards) / len(rewards)
        # print("reward = ",avg_reward)


        if avg_reward > -100:
            total_rewards.append(avg_reward)
            total_latency.append(sum(latencies)/len(latencies))
            access_rate.append(np.mean(episode_access_rate))
            epsilon_values.append(dqn.epsilon)
            latency_deviation.append(np.mean(deviations))

            count_explore = exploit_or_explore.count("explore")
            # print "count_explore : ",count_explore
            # print "length of explore or exploit : ",len(exploit_or_explore)
            if count_explore > (len(exploit_or_explore) / 2):
                total_exploit_or_explore.append("explore")
            else:
                total_exploit_or_explore.append("exploit")

        print("episode: {}/{}, score: {}, average latency: {}, required_latency: {}, e: {},"
                    .format(episode+1, NO_EPISODES, np.mean(rewards), np.mean(latencies),REQUIRED_LATENCY,dqn.epsilon))




    episode_access_rate  = []
    exploit_or_explore = []
    avg_waiting_time = [0]
    deviations = []
    latencies = []
    rewards = []
    # print("time taken for the episode is: ",time.time() - t3)
    # print("\n")
dqn.model.save("DQN_Model.h5")

temp_rewards = [np.mean(total_rewards[i:i+25]) for i in range(0,len(total_rewards),25)]
temp_latencies = [np.mean(total_latency[i:i+25]) for i in range(0,len(total_latency),25)]
temp_loss = [np.mean(dqn.loss[i:i+25]) for i in range(0,len(dqn.loss),25)]
temp_val_loss = [np.mean(dqn.val_loss[i:i+25]) for i in range(0,len(dqn.val_loss),25)]
temp_deviation = [np.mean(latency_deviation[i:i+25]) for i in range(0,len(latency_deviation),25)]
temp_epsilon = [np.mean(epsilon_values[i:i+25]) for i in range(0,len(epsilon_values),25)]
temp_access_rate = [np.mean(access_rate[i:i+25]) for i in range(0,len(access_rate),25)] 

final_df = pd.DataFrame(columns=["total_rewards","total_latency","latency_deviation","epsilon","Access_rate"])
final_df = {}

final_df["total_rewards"] = total_rewards
final_df["total_latency"] = total_latency
final_df["latency_deviation"] = latency_deviation
final_df["epsilon_values"] = epsilon_values
final_df["access_rate"] = access_rate
final_df["explore/exploit"] = total_exploit_or_explore

f = plt.figure(1)

# print(total_rewards)
plt.plot(np.arange(len(temp_rewards)), temp_rewards)

# f.show()

f.savefig("dql_episodic_rewards.png")


g = plt.figure(2)

plt.plot(np.arange(len(temp_latencies)), temp_latencies)

# g.show()

g.savefig("dql_episodic_latency.png")

h = plt.figure(3)

plt.plot(np.arange(len(temp_loss)), temp_loss)

# h.show()

h.savefig("dql_loss.png")

i = plt.figure(4)

plt.plot(np.arange(len(temp_epsilon)), temp_epsilon)

# i.show()

i.savefig("epsilon values.png")

j = plt.figure(5)

plt.plot(np.arange(len(temp_deviation)), temp_deviation)

# j.show()

j.savefig("latency deviation.png")

k = plt.figure(6)

plt.plot(np.arange(len(temp_access_rate)), temp_access_rate)

# j.show()

k.savefig("access rate variation.png")

l = plt.figure(7)

plt.plot(np.arange(len(temp_val_loss)), temp_val_loss)

l.savefig("dql_val_loss.png")

save_df = pd.DataFrame.from_dict(final_df)

save_df.to_csv("Obtained Results.csv")



