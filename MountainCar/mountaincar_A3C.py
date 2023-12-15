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
    hyperparam_storage_dir = 'D:\\Artificial_Intelligence\\Portfolio\\RL_updated\\MountainCar\\Hyperparam_tuning_a3c\\' # Windows
else:
    hyperparam_storage_dir = '/media/n/NewDisk/Artificial_Intelligence/Portfolio/RL_updated/MountainCar/Hyperparam_tuning_a3c/' # Linux /media/near/2E950FA76FED20DC/Artificial_Intelligence/Portfolio/RL_updated/MountainCar

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_visible_devices([], 'GPU')

seed =0
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_env', type=int, required=True)
parser.add_argument('--discount', type=float, required=True)
parser.add_argument('--end_of_episode', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--entropy_fact', type=float, required=True)
parser.add_argument('--ep', type=float, required=True)
parser.add_argument('--bolt_fact', type=float, required=True)
parser.add_argument('--exploration_tech', type=str, required=True)
parser.add_argument('--train_steps', type=int, required=True)
parser.add_argument('--file_name', type=str, required=True)
parser.add_argument('--dense_units', type=int, required=True)
parser.add_argument('--time_to_update', type=int, required=True)
parser.add_argument('--use_LSTM', type=int, required=True)




env = gym.make("MountainCar-v0")#,new_step_api=True
n_actions = env.action_space.n
state_space_samples = np.array([env.observation_space.sample() for x in range(50000)])
scaler = sklearn.preprocessing.StandardScaler()
exploration_technique = ""

#function to normalize states
def scale_state(state):                  #requires input shape=(2,)
    scaled = scaler.transform(state)
    return scaled                        #returns shape =(1,2)   


class ActorCriticAgent(tf.keras.Model):
    def __init__(self, state_shape, n_actions, nEnviroment, end_of_episode, d, lstm_unit,  use_LSTM=False, activation = "relu"):
        super(ActorCriticAgent, self).__init__()
        self.n_actions = n_actions
        self.nEnviroment = nEnviroment
        self.d0 = Dense(d, activation=activation, name ="inp0")
        self.Qvalues = Dense(self.n_actions, activation='linear', name ="logits")
        self.d_state_value = Dense(1, activation='linear', name ="v")
        self.end_of_episode = end_of_episode
        self.use_LSTM = use_LSTM
        self.lstm = LSTM(lstm_unit, return_sequences=False, return_state = False)
        self.state_buffer = None

    def call(self, observations, inference= True):
        
        if self.use_LSTM:
            x = self.lstm(observations)
        else:
            x = observations

        x = self.d0(x)
        logits = self.Qvalues(x)
        state_value = self.d_state_value(x)

        return logits, state_value[:,0]



def get_valid_trials_number(hyper_dir):
    
    dir = "./"+hyper_dir
    trial_n = 0 
    if os.path.isdir(dir) :
        trial_n = len(next(os.walk(dir))[1])
        print("Trial number : ",trial_n)
        
    return str(trial_n)+"_"
class A3CAgent_Optimization(tf.keras.Model):
    def __init__(self, lr, entropy_factor, exploration_tech, 
    discount, time_to_update, dense_units,  lstm_units, n_enviroment, writer,  use_LSTM = False, trial_n = "", end_of_episode = 200, evaluation_epoch = 2500, scale_state = scaler):
        super(A3CAgent_Optimization, self).__init__()
        # Enviroment parameters
        self.env = gym.make("MountainCar-v0").env
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n 
        self.evaluation_epoch = evaluation_epoch
        
        
        # Training parameters
        self.end_of_episode = end_of_episode

        self.n_enviroment = n_enviroment
        self.learning_rate= lr
        self.entropy_factor = entropy_factor
        self.exploration_tech = exploration_tech
        self.discount = discount
        self.time_to_update = time_to_update
        self.epsilon = 1
        self.boltzman_factor = 1
        self.optimizer=Adam(learning_rate=lr)
        self.use_LSTM= use_LSTM
        self.scale_state = scale_state
        self.use_scale = False

        self.actorcritic_agent = ActorCriticAgent(
                    state_shape = obs_shape,
                    n_actions = n_actions, 
                    nEnviroment = n_enviroment,
                    end_of_episode =end_of_episode,#200,
                    d = dense_units,
                    lstm_unit = lstm_units,
                    use_LSTM = use_LSTM
                    )
        
        self.train_env =  gym.make("MountainCar-v0").env 
        
        # Logs parameters
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('n_enviroment', n_enviroment, step=0)
            tf.summary.scalar('learning_rate', self.learning_rate, step=0)
            tf.summary.scalar('entropy_factor', self.entropy_factor, step=0)
            tf.summary.scalar('discount', discount, step=0)
            tf.summary.scalar('dense_units', dense_units, step=0)
            tf.summary.scalar('time_to_update', time_to_update, step=0)
            tf.summary.text('exploration_technique', exploration_tech, step=0)
            tf.summary.scalar('use_LSTM', int(self.use_LSTM), step=0)
            tf.summary.scalar('end_of_episode', self.end_of_episode , step=0)

        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewards = 0

        self.train_envs_list = [gym.make("MountainCar-v0").env  for _ in range(n_enviroment)]#
        self.iter = [0 for _ in range(n_enviroment)]
        
        self.buffer_size = 4
        if self.use_LSTM:
            self.memory = [deque(maxlen= self.buffer_size) for _ in range(n_enviroment)]
            self.new_memory = [deque(maxlen= self.buffer_size) for _ in range(n_enviroment)]
            self.eval_memory = deque(maxlen= self.buffer_size)
       
    def env_reset(self):
        self.iter=[0 for _ in range(self.n_enviroment)]
        return np.array([env_x.reset()[0] for env_x in self.train_envs_list])
  

    def env_step(self, actions):
            
        results = [env_x.step(a) for env_x, a in zip(self.train_envs_list, actions)]

        new_obs, rewards, terminated, truncated , infos  = map(np.array, zip(*results))
        done = np.zeros((self.n_enviroment,1))
        
        for i in range(self.n_enviroment):
            self.iter[i] +=1
            done[i] = terminated[i] or truncated[i]

            if self.iter[i] == self.end_of_episode or  done[i][0] == 1:
                
                new_obs[i] = self.train_envs_list[i].reset()[0]
                done[i][0] = 1
                self.iter[i] = 0

                if self.use_LSTM:
                    self.memory[i].clear()
                    self.new_memory[i].clear()
            
            done[i] = 1-int(done[i])    
        return new_obs, rewards, done[:,0], infos
    
    def fill_lstm_buffer(self, obs, is_new_obs = False, eval= False):
        
        if eval:
            buffer = np.zeros((1, self.buffer_size, self.obs_shape[0]))
            self.eval_memory.append(obs[0])
            padding = self.buffer_size - len(self.eval_memory)
            buffer[0,:,:] = np.array(list(self.eval_memory) + [np.zeros(self.obs_shape, dtype= "float32") for _ in range(padding)])
            return buffer

        buffer = np.zeros((self.n_enviroment, self.buffer_size, self.obs_shape[0]))
        for i in  range(self.n_enviroment):
            
            if is_new_obs:
                self.new_memory[i].append(obs[i])
                padding = self.buffer_size - len(self.new_memory[i])
                buffer[i,:,:] = np.array(list(self.new_memory[i]) + [np.zeros(self.obs_shape, dtype= "float32") for _ in range(padding)])
                
            else:
                self.memory[i].append(obs[i])
                padding = self.buffer_size - len(self.memory[i])
                buffer[i,:,:] = np.array(list(self.memory[i]) + [np.zeros(self.obs_shape, dtype= "float32") for _ in range(padding)])
        return buffer


   
    def run_and_train_agent(self, epoch, obs):

        with tf.GradientTape() as tape:

            if self.use_LSTM: 
                original_obs = obs
                obs = self.fill_lstm_buffer(obs)

            logits, state_values= self.actorcritic_agent(obs)
            actions, random_choice = self.sample_actions(logits, self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
            new_obs, rewards, done, _ = self.env_step(actions)

            if self.use_LSTM: 
                original_new_obs = new_obs
                new_obs = self.fill_lstm_buffer(new_obs,is_new_obs = True)

            _, next_state_values = self.actorcritic_agent(new_obs)
            next_state_values = next_state_values*done
            target_state_values = rewards + tf.math.multiply(self.discount,next_state_values)
            advantage = target_state_values - state_values
            
            probs = tf.nn.softmax(logits, axis=-1)            # [n_envs, n_actions]
            logprobs = tf.nn.log_softmax(logits, axis=-1)     # [n_envs, n_actions]
            logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions, self.n_actions), axis=-1) # [n_envs,]

            entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")
            critic_loss = 0.5*tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2, axis=0)
            actor_loss = -tf.reduce_mean(logp_actions * tf.stop_gradient(advantage), axis=0) - self.entropy_factor * tf.reduce_mean(entropy, axis=0)
            loss = actor_loss + critic_loss

        variables = self.actorcritic_agent.trainable_variables 
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.total_rewards = self.total_rewards + rewards[0]

        if epoch % self.time_to_update == 0: self.update_exploration_factor()
        
        if self.use_LSTM: 
            obs= original_obs
        self.append_training_metrics(actions, obs, done, random_choice, entropy,actor_loss,critic_loss,loss, advantage,target_state_values,state_values, epoch)

        if (done[0] == 0) and epoch >0  :
            self.rewards_train_history.append(self.total_rewards)
        
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Training_rewards', self.total_rewards, step=epoch)

            self.total_rewards = 0
        if self.use_LSTM:
            return original_new_obs
        else:
            return new_obs
    
    # def run_and_train_agent(self, epoch, obs):
    #     print("initis ...",obs.shape, obs)
    #     if self.use_LSTM: obs = self.fill_lstm_buffer(obs)
    #     logits, _= self.actorcritic_agent(obs)
    #     actions, random_choice = self.sample_actions(logits, self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
    #     new_obs, rewards, done, _ = self.env_step(actions)
    #     if self.use_LSTM: new_obs = self.fill_lstm_buffer(new_obs)
        
    #     with tf.GradientTape() as tape:
            
    #         logits, state_values= self.actorcritic_agent(obs)
    #         _, next_state_values = self.actorcritic_agent(new_obs)
    #         next_state_values = next_state_values*done

    #         target_state_values = rewards + tf.math.multiply(self.discount,next_state_values)
    #         advantage = target_state_values - state_values
            
    #         probs = tf.nn.softmax(logits, axis=-1)            # [n_envs, n_actions]
    #         logprobs = tf.nn.log_softmax(logits, axis=-1)     # [n_envs, n_actions]
    #         logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions, self.n_actions), axis=-1) # [n_envs,]

    #         entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")
    #         critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2, axis=0)
    #         actor_loss = -tf.reduce_mean(logp_actions * tf.stop_gradient(advantage), axis=0) - self.entropy_factor * tf.reduce_mean(entropy, axis=0)
    #         loss = actor_loss + critic_loss

    #     variables = self.actorcritic_agent.trainable_variables 
    #     gradients = tape.gradient(loss, variables)
    #     self.optimizer.apply_gradients(zip(gradients, variables))

    #     self.total_rewards = self.total_rewards + rewards[0]

    #     if epoch % self.time_to_update == 0: self.update_exploration_factor()
        
    #     if self.use_LSTM: obs = obs[:,-1,:]
    #     print(obs.shape, obs)
    #     self.append_training_metrics(actions, obs, done, random_choice, entropy,actor_loss,critic_loss,loss, advantage,target_state_values,state_values, epoch)

    #     if (done[0] == 0) and epoch >0  :
    #         self.rewards_train_history.append(self.total_rewards)
        
    #         with self.tb_summary_writer.as_default():
    #             tf.summary.scalar('Training_rewards', self.total_rewards, step=epoch)

    #         self.total_rewards = 0

    #     return new_obs


    def sample_actions(self, logits, epsilon, boltzman_factor=1,  exploration = "soft", inference = False, print_policy = False):

        if inference:
            policy = tf.stop_gradient(tf.nn.softmax(logits, axis=-1)).numpy()
            return np.array([np.argmax(p.reshape((1,p.shape[0])), axis = 1) for p in policy])[:,0], 4
        
        policy = tf.stop_gradient(tf.nn.softmax(logits/boltzman_factor, axis=-1)).numpy()
        if exploration == "epsilon":
            rand = np.random.random()
            if rand < epsilon:
                return np.random.randint(self.n_actions, size=len(policy), dtype = np.int64), 1
            
            return  np.array([np.random.choice(len(p), p=p) for p in policy]), 2 #np.array([np.argmax(p.reshape((1,p.shape[0])), axis = 1) for p in policy])[:,0], 2

        if exploration == "boltzman":
            return np.array([np.random.choice(len(p), p=p) for p in policy]), 3 #np.array([np.argmax(p.reshape((1,p.shape[0])), axis = 1)  for p in policy])[:,0], 3
        
        return np.array([np.random.choice(len(p), p=p) for p in policy]), 0
       
    

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()[0]
            if self.use_LSTM: self.eval_memory.clear()
            total_reward = 0
            for it in range(200):#
                
                # if self.use_scale :
                #     obs = self.scale_state.transform(obs.reshape((1,self.obs_shape[0])))
                # else:
                obs = obs.reshape((1,self.obs_shape[0]))

                if self.use_LSTM:
                    obs = self.fill_lstm_buffer(obs, eval= True)
                    
                logit, _= self.actorcritic_agent(obs, inference= True)

                if hyp : logit.numpy()
                
                action,_ = self.sample_actions(logit,0, boltzman_factor=1, inference= True)
                action = action[0]
                obs, reward, terminated, truncated , info = eval_env.step(action)
                total_reward += reward

                done = truncated or terminated 
                if done:
                    break

            rewards_history.append(total_reward)

        return rewards_history


    def evaluate_agent(self, epoch):
        if (epoch %  self.evaluation_epoch == 0 )  and epoch >0:
            rewards_history = np.mean(self.evaluate(self.env , n_tries=3))
            self.rewards_val_history.append(rewards_history)

            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', self.rewards_val_history[-1], step=epoch)

                if len(self.rewards_val_history)> 100: 
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-100:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-100:]), step=epoch)


    def append_training_metrics(self, _batch_actions, _batch_states, _batch_done,rand_, _entropy,_actor_loss,_critic_loss,_loss,_advantage,_target_state_values,_state_values, epoch):

        if _batch_done[0] == 0:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('RT_done', 0, step=epoch)
                tf.summary.scalar('RT_done', 1, step=epoch+1)

        if epoch %50 == 0:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('RT_rewards', self.total_rewards, step=epoch)
                tf.summary.scalar('actor_loss', _actor_loss, step=epoch)
                tf.summary.scalar('critic_loss', _critic_loss, step=epoch)
                tf.summary.scalar('entropy_history', tf.reduce_mean(tf.convert_to_tensor(_entropy), axis = 0), step=epoch)
                tf.summary.scalar('Target_history', tf.reduce_mean(tf.convert_to_tensor(_target_state_values), axis = 0), step=epoch)
                tf.summary.scalar('v_history', tf.reduce_mean(tf.convert_to_tensor(_state_values), axis = 0), step=epoch)
                tf.summary.scalar('advantage_history', tf.reduce_mean(tf.convert_to_tensor(_advantage), axis = 0), step=epoch)

                tf.summary.scalar("RT_state", _batch_states[0][0], step=epoch)
                tf.summary.scalar("RT_velocity", _batch_states[0][1], step=epoch)
                tf.summary.scalar('action', _batch_actions[0], step=epoch)
                

                tf.summary.scalar('random_choise', rand_, step=epoch)
                tf.summary.scalar('RT_epsilon', self.epsilon, step=epoch)
                tf.summary.scalar('RT_boltzman_factor', self.boltzman_factor, step=epoch)


    def update_exploration_factor(self):
        if self.exploration_tech == "epsilon" and self.epsilon > 0.0001 :
            self.epsilon = self.epsilon*0.99
            
        elif self.exploration_tech == "boltzman" and self.boltzman_factor > 0.01 :
            self.boltzman_factor = self.boltzman_factor*0.99
        

            
        

def main(n_enviroment, discount, end_of_episode, learning_rate, entropy_factor, epsilon, boltzman_factor, exploration_tech, train_steps, file_name, time_to_update, ulsm ):


    writer = "Manual_hyper/fit_a3c/"
    model = A3CAgent_Optimization(
        lr = learning_rate, 
        entropy_factor = entropy_factor,
        exploration_tech = exploration_tech,
        discount = discount, 
        time_to_update = time_to_update, 
        dense_units= dense_units, 
        lstm_units = 32,#lstm_units,
        n_enviroment = n_enviroment,
        writer = writer,
        trial_n = get_valid_trials_number(writer),
        use_LSTM=ulsm,
        end_of_episode = end_of_episode
 )

    obs = model.env_reset()

    print("Starting "+file_name+"...")
    for epoch in range(train_steps):
        
        new_obs = model.run_and_train_agent(epoch, obs)
        model.evaluate_agent(epoch)

        obs = new_obs

        if len(model.rewards_val_history)> 100 :
            if np.mean(model.rewards_val_history[-100:]) >= -100:
                print("Your agent reached the objetive")
                break

    results = {}
    results["final_epsilon"] = ep
    results["final_boltzman"] = bolt_fact
    results["rewards"]= model.rewards_val_history
    results["mean_rewards"]= np.mean(model.rewards_val_history)

    with open(hyperparam_storage_dir+file_name+'.json', 'w') as convert_file:
        convert_file.write(json.dumps(results))

    print("Finishing "+file_name+"...")


if __name__ == "__main__":

    args = parser.parse_args()
    n_env = args.n_env#[0]
    discount =args.discount#[1]
    end_of_episode = args.end_of_episode#[2]
    lr = args.lr#[3]
    entropy_fact = args.entropy_fact#[4]
    ep = args.ep#[5]
    bolt_fact = args.bolt_fact#[6]
    exploration_tech = args.exploration_tech#[7]
    train_steps = args.train_steps#[8]
    file_name = args.file_name#[9]
    dense_units = args.dense_units#[10]
    time_to_update = args.time_to_update#[10]
    ulsm= bool(args.use_LSTM)#[10]

    print("n_env : ",n_env)
    print("discount : ",discount)
    print("end_of_episode : ",end_of_episode)
    print("lr : ",lr)
    print("entropy_fact : ",entropy_fact)
    print("ep : ",ep)
    print("bolt_fact : ",bolt_fact)
    print("exploration_tech : ",exploration_tech)
    print("train_steps : ",train_steps)
    print("file_name : ",file_name)
    print("dense_units : ",dense_units)
    print("time_to_update : ",time_to_update)
    print("use_LSTM : ",ulsm)
    
    

    exploration_technique = exploration_tech

    # scaler.fit(state_space_samples)

    main(n_env, discount, end_of_episode, lr, entropy_fact, ep, bolt_fact, exploration_tech, train_steps, file_name, time_to_update, ulsm)
    print("--------------------------------\n\n")
    
