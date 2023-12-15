import numpy as np
import tensorflow as tf


import matplotlib.pyplot as plt

import gym
from IPython import display as ipythondisplay
from IPython.display import clear_output
from pyvirtualdisplay import Display

from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input , BatchNormalization, Concatenate, LayerNormalization, LSTM,Reshape
from tensorflow.keras.optimizers import Adam
import random
from livelossplot.inputs.tf_keras import PlotLossesCallback
from datetime import datetime

import tqdm
import pandas as pd

import math
import time

from livelossplot import PlotLosses
import pickle

from multiprocessing import Pool, freeze_support
import itertools
import json


import sklearn
import sklearn.preprocessing
import os 

if os.name == 'nt':
    hyperparam_storage_dir = 'D:\\Artificial_Intelligence\\Portfolio\\RL_updated\\MountainCar\\Hyperparam_tuning_ddqn\\' # Windows
else:
    hyperparam_storage_dir = '/media/n/NewDisk/Artificial_Intelligence/Portfolio/RL_updated/MountainCar/Hyperparam_tuning_ddqn/' # Linux

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_visible_devices([], 'GPU')

seed =0
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--discount', type=float, required=True)
parser.add_argument('--end_of_episode', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--tau_update_network', type=float, required=True)
parser.add_argument('--exploration_tech', type=str, required=True)
parser.add_argument('--train_steps', type=int, required=True)
parser.add_argument('--file_name', type=str, required=True)
parser.add_argument('--dense_units', type=int, required=True)
parser.add_argument('--time_to_update', type=int, required=True)


env = gym.make("MountainCar-v0")#,new_step_api=True
n_actions = env.action_space.n
state_space_samples = np.array([env.observation_space.sample() for x in range(50000)])
scaler = sklearn.preprocessing.StandardScaler()
exploration_technique = ""

#function to normalize states
def scale_state(state):                  #requires input shape=(2,)
    scaled = scaler.transform(state)
    return scaled                        #returns shape =(1,2)   

def get_valid_trials_number(hyper_dir):
    
    dir = "./"+hyper_dir
    trial_n = 0 
    if os.path.isdir(dir) :
        trial_n = len(next(os.walk(dir))[1])
        print("Trial number : ",trial_n)
        
    return str(trial_n)+"_"

class DQNAgent(tf.keras.Model):
    def __init__(self, state_shape, n_actions, d):
        super(DQNAgent, self).__init__()
        self.n_actions = n_actions
        self.d0 = Dense(d, activation='relu', name ="inp0")
        self.Qvalues = Dense(self.n_actions, activation='linear', name ="logits")

    def call(self, observations):
        
        x = observations
        x = self.d0(x)
        logits = self.Qvalues(x)

        return logits
  
  

class DDQNAgent_Optimization(tf.keras.Model):
    def __init__(self, lr, discount, time_to_update, dense_units, tau_update_network, writer, trial_n, end_of_episode =1000, evaluation_epoch = 1000, training_epoch = 10,  batch_size = 256, exploration_tech = "boltzman" ,  scale_state = scaler):
        super(DDQNAgent_Optimization, self).__init__()

        # Enviroment parameters
        self.env = gym.make("MountainCar-v0").env
        self.train_env = gym.make("MountainCar-v0").env
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n 
        self.scale_state = scale_state
        self.use_scale = False
        self.episode_steps = 0 
        self.evaluation_epoch = evaluation_epoch
        self.training_epoch = training_epoch

        
        # Training parameters
        self.batch_size = batch_size
        self.end_of_episode = end_of_episode
        self.memory = deque(maxlen= 10000)

        self.writer = writer
        self.tau_update_network = tau_update_network
        self.learning_rate= lr
        self.discount = discount
        self.time_to_update = time_to_update
        self.epsilon = 1
        self.boltzman_factor = 1
        self.optimizer=Adam(learning_rate=lr)
        self.exploration_tech = exploration_tech
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.DQN_agent = DQNAgent(
                    state_shape = obs_shape,
                    n_actions = n_actions, 
                    d = dense_units,
                    )

        self.Target_DQN_agent = DQNAgent(
                    state_shape = self.obs_shape,
                    n_actions = self.n_actions, 
                    d = dense_units,
                    )

        # Logs parameters   
            
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('learning_rate', self.learning_rate, step=0)
            tf.summary.scalar('discount', self.discount, step=0)
            tf.summary.scalar('dense_units', dense_units, step=0)
            tf.summary.scalar('time_to_update', time_to_update, step=0)
            tf.summary.scalar('tau_update_network', tau_update_network, step=0)
            tf.summary.scalar('end_of_episode', end_of_episode , step=0)
            tf.summary.text('exploration_technique', exploration_tech, step=0)

        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewards = 0


    def sample_actions(self, q_values, epsilon, boltzman_factor = 1, exploration = "epsilon", inference = False):


        if exploration == "boltzman":
            q_values = tf.stop_gradient(tf.nn.softmax(q_values / boltzman_factor, axis=-1)).numpy()   
            best_actions = np.argmax(q_values, axis = -1) 
        else: 
            best_actions = np.argmax(q_values, axis = -1) 

        if inference:
            return best_actions, 2
        
        if exploration == "epsilon":
            rand = np.random.random()
            if rand < epsilon:
                return np.random.randint(self.n_actions, size=len(q_values), dtype = np.int64), 0
            
        return best_actions, 1
       
  
    def evaluate(self, eval_env, n_tries=1):
        rewards_history = []
        for _ in range(n_tries):
            state = eval_env.reset()[0]
            total_reward = 0
            for it in range(200):#
                # if self.use_scale : 
                #     current_qvalue = self.DQN_agent(self.scale_state.transform((state.reshape((1,len(state))))))
                # else:
                current_qvalue = self.DQN_agent(state.reshape((1,len(state))))

                # current_qvalue =  self.DQN_agent((state.reshape((1,len(state)))))
                action,_ = self.sample_actions(current_qvalue,0, 1, exploration="soft", inference= True)
                action = action[0]
                state, reward, terminated, truncated , info = eval_env.step(action)
                total_reward += reward

                done = truncated or terminated 
                if done:
                    break

            rewards_history.append(total_reward)

        return rewards_history


    def get_sample(self):
        batch = random.sample(self.memory, self.batch_size)

        s = np.array([each[0] for each in batch])
        a = [each[1] for each in batch]
        s_ = np.array([each[2] for each in batch])
        r = [each[3] for each in batch]
        dones = [(1-each[4]) for each in batch]
        return s,a,r,s_,dones


    def run_agent(self, epoch, obs):

        self.episode_steps +=1 

        # if self.use_scale : 
        #     current_qvalue = self.DQN_agent(self.scale_state.transform(obs.reshape((1,len(obs)))))
        # else:
        current_qvalue = self.DQN_agent(obs.reshape((1,len(obs))))

        actions, random_selection = self.sample_actions(current_qvalue,  self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
        actions = actions[0]
        new_obs, rewards, terminated, truncated , infos= self.train_env.step(actions)
        self.total_rewards = self.total_rewards  + rewards
        done = terminated or truncated

        if self.episode_steps %  self.end_of_episode == 0 and epoch >0: done = True
        
        self.append_sample_metrics(epoch, obs, actions, done, random_selection)

        # if self.use_scale:
        #     self.memory.append([self.scale_state.transform(obs.reshape((1,len(obs))))[0], actions, self.scale_state.transform(new_obs.reshape((1,len(new_obs))))[0], rewards, done])
        # else:
        self.memory.append([obs, actions, new_obs, rewards, done])

        obs = new_obs
        if done and epoch>0 : 
            self.rewards_train_history.append(self.total_rewards)
            
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Training_rewards', self.total_rewards, step= len(self.rewards_train_history) )
                tf.summary.scalar('RT_done', 0, step=epoch)
                tf.summary.scalar('RT_done', 1, step=epoch+1)

            self.total_rewards = 0
            obs = self.train_env.reset()[0]  
            self.episode_steps = 0 

        return obs

    def train_agent(self, epoch):
        if  epoch % self.training_epoch == 0  and len(self.memory)>= self.batch_size: #
            obs_batch, action_batch, reward_batch, new_obs_batch, done_batch = self.get_sample()
            
            with tf.GradientTape() as tape:
                current_qvalue_batch = self.DQN_agent(obs_batch)
                next_qvalue_batch = self.Target_DQN_agent(new_obs_batch)

                next_qvalue_batch = tf.cast(next_qvalue_batch, dtype= tf.float32)
                current_qvalue_batch = tf.cast(current_qvalue_batch, dtype= tf.float32)
                done_batch = tf.cast(done_batch, dtype= tf.float32)
                reward_batch = tf.cast(reward_batch, dtype= tf.float32)
                
                current_action_qvalues = tf.reduce_sum(tf.one_hot(action_batch, self.n_actions) * current_qvalue_batch, axis=1)
                next_state_values_target = tf.reduce_max(next_qvalue_batch, axis=1)# if it was sarsa, we would select the state value related to A' (with next action Index) instead of getting the max
                
                target_state_values = reward_batch + tf.math.multiply(self.discount,(done_batch)*next_state_values_target)
                loss = self.loss_fn(target_state_values, current_action_qvalues)
            
            variables = self.DQN_agent.trainable_variables 
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            self.append_training_metrics(action_batch, obs_batch, loss, target_state_values,next_qvalue_batch,current_qvalue_batch, epoch)


    def update_training_params(self, epoch):

        if epoch% self.time_to_update == 0 and self.exploration_tech == "epsilon" and self.epsilon > 0.0001 :
            self.epsilon = self.epsilon*0.99
            
        elif epoch% self.time_to_update == 0 and self.exploration_tech == "boltzman" and self.boltzman_factor > 0.01 :
            self.boltzman_factor = self.boltzman_factor*0.99

        
        self.load_weigths_into_target_network()

    def evaluate_agent(self, epoch):
        if epoch %self.evaluation_epoch ==0 and epoch >0:
            rewards_history = np.mean(self.evaluate(self.env , n_tries=3))
            self.rewards_val_history.append(rewards_history)

            
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', rewards_history, step=int(epoch/self.end_of_episode) )
                
                if len(self.rewards_val_history)> 100: 
                    # print(f"Eval average is {np.mean(self.rewards_val_history[-100:]) } . Training average is {np.mean(self.rewards_train_history[-100:])}")
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-100:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-100:]), step=epoch)

   
    def append_sample_metrics(self, epoch, obs, actions, done, random_selection):


        if epoch %50 == 0:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('RT_action', actions, step=epoch)
                tf.summary.scalar("RT_state", obs[0], step=epoch)
                tf.summary.scalar("RT_velocity", obs[1], step=epoch)
                tf.summary.scalar("RT_epsilon", self.epsilon, step=epoch)
                tf.summary.scalar('RT_boltzman_factor', self.boltzman_factor, step=epoch)
                tf.summary.scalar('RT_rewards', self.total_rewards, step=epoch)
                tf.summary.scalar('random_selection', random_selection, step=epoch)

    def append_training_metrics(self,  _batch_actions, _batch_states,  loss,_target_state_values, next_qvalue_batch, current_qvalue_batch, epoch):

        if epoch %50 == 0:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Loss', loss, step=epoch)
                tf.summary.scalar('Target_history', tf.reduce_mean(tf.convert_to_tensor(_target_state_values), axis = 0), step=epoch)
                tf.summary.scalar('Training_action', _batch_actions[0], step=epoch)
                tf.summary.scalar("Training_state", _batch_states[0][0], step=epoch)
                tf.summary.scalar("Training_velocity", _batch_states[0][1], step=epoch)

    def load_weigths_into_target_network(self):

        """ assign target_network.weights variables to their respective agent.weights values. """
        
        for w_agent, w_target in zip(self.DQN_agent.trainable_weights, self.Target_DQN_agent.trainable_weights):
            new_w_agent = self.tau_update_network*w_agent + (1 - self.tau_update_network) * w_target
            w_target.assign(new_w_agent)



def main(discount, end_of_episode, learning_rate, tau_update_network, exploration_tech, train_steps, file_name, time_to_update):


    writer = "Manual_hyper/fitDDQN/"
    model = DDQNAgent_Optimization(
        lr = learning_rate, 
        discount = discount, 
        time_to_update = time_to_update, 
        dense_units= dense_units, 
        tau_update_network = tau_update_network,
        writer = writer ,
        end_of_episode = end_of_episode,
        exploration_tech = exploration_tech,
        trial_n = get_valid_trials_number(writer))
    

    obs = model.train_env.reset()[0]

    print("Starting "+file_name+"...")
    for epoch in range(train_steps):
        
        obs = model.run_agent(epoch, obs)
        model.train_agent(epoch)               
        model.update_training_params(epoch)
        model.evaluate_agent(epoch)

        
        if len(model.rewards_val_history)> 100 :
            if np.mean(model.rewards_val_history[-100:]) >= -100:
                print("Your agent reached the objetive")
                break

    results = {}
    results["rewards"]= model.rewards_val_history
    results["mean_rewards"]= np.mean(model.rewards_val_history)

    with open(hyperparam_storage_dir+file_name+'.json', 'w') as convert_file:
        convert_file.write(json.dumps(results))

    print("Finishing "+file_name+"...")


if __name__ == "__main__":

    args = parser.parse_args()
    discount =args.discount#[1]
    end_of_episode = args.end_of_episode#[2]
    lr = args.lr#[3]
    tau_update_network = args.tau_update_network#[4]
    exploration_tech = args.exploration_tech#[7]
    train_steps = args.train_steps#[8]
    file_name = args.file_name#[9]
    dense_units = args.dense_units#[10]
    time_to_update = args.time_to_update#[10]

    print("discount : ",discount)
    print("end_of_episode : ",end_of_episode)
    print("lr : ",lr)
    print("tau_update_network : ",tau_update_network)
    print("exploration_tech : ",exploration_tech)
    print("train_steps : ",train_steps)
    print("file_name : ",file_name)
    print("dense_units : ",dense_units)
    print("time_to_update : ",time_to_update)
    

    exploration_technique = exploration_tech

    scaler.fit(state_space_samples)

    main(discount, end_of_episode, lr, tau_update_network,  exploration_tech, train_steps, file_name, time_to_update)
    print("--------------------------------\n\n")
    
