# %load_ext tensorboard
import datetime

import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import gym
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import random
import tqdm
import time
import itertools
import shutil
import json
import sklearn
import sklearn.preprocessing
import keras_tuner as kt
import tensorflow_probability as tfp


import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder


print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)




class ContinuousA3CAgent_Optimization(tf.keras.Model):
    def __init__(self, lr, entropy_factor,
    discount, dense_units,  lstm_units, n_enviroment, writer,  use_LSTM = False, trial_n = "", end_of_episode = 200, 
    evaluation_epoch = 2500, environment_name="MountainCar-v0", sigma_noise= 1e-5, reward_scaler  = 1):
        super(ContinuousA3CAgent_Optimization, self).__init__()
        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        obs_shape = self.env.observation_space.shape
        self.evaluation_epoch = evaluation_epoch
        self.environment_name= environment_name
        
        
        # Training parameters
        self.end_of_episode = end_of_episode

        self.n_enviroment = n_enviroment
        self.learning_rate= lr
        self.entropy_factor = entropy_factor
        self.sigma_noise = sigma_noise
        self.discount = discount
        self.epsilon = 1
        self.boltzman_factor = 1
        self.optimizer=Adam(learning_rate=lr)
        self.use_LSTM= use_LSTM
        self.reward_scaler = reward_scaler


        self.actorcritic_agent = ContinuousActorCriticAgent(
                    env = self.env, 
                    d = dense_units,
                    lstm_unit = lstm_units,
                    use_LSTM = use_LSTM,
                    sigma_noise = self.sigma_noise
                    )
        
        self.train_env =  gym.make(environment_name).env 
        
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
            tf.summary.scalar('use_LSTM', int(self.use_LSTM), step=0)
            tf.summary.scalar('end_of_episode', self.end_of_episode , step=0)
            tf.summary.scalar('sigma_noise', self.sigma_noise , step=0)

        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewards = 0

        self.train_envs_list = [gym.make(environment_name).env  for _ in range(n_enviroment)]#
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

            
            actions, state_values, mu, sigma = self.actorcritic_agent(obs)

            if self.environment_name in ["BipedalWalker-v3"]:
                new_obs, rewards, done, _ = self.env_step(actions)
            else:
                new_obs, rewards, done, _ = self.env_step(tf.expand_dims(actions, axis = 1))

            rewards = rewards/self.reward_scaler

            if self.use_LSTM: 
                original_new_obs = new_obs
                new_obs = self.fill_lstm_buffer(new_obs,is_new_obs = True)

            _, next_state_values, _, _ = self.actorcritic_agent(new_obs)

            next_state_values = next_state_values*done
            target_state_values = rewards + tf.math.multiply(self.discount,next_state_values)
            advantage = target_state_values - state_values
            
            normal_dist = tfp.distributions.Normal(mu, sigma)
            entropy = normal_dist.entropy() 
            if self.environment_name in ["BipedalWalker-v3"]:
                entropy = tf.reduce_mean(entropy, axis=1)
                actor_loss = tf.reduce_mean(-tf.reduce_mean(normal_dist.log_prob(actions) * tf.stop_gradient(tf.expand_dims(advantage, axis = 1)), axis=1)  - self.entropy_factor * tf.reduce_mean(entropy, axis=0), axis=0)
                actions = actions[:,0]
            else:
                actor_loss = -tf.reduce_mean(normal_dist.log_prob(actions) * tf.stop_gradient(advantage), axis=0) - self.entropy_factor * tf.reduce_mean(entropy, axis=0)

            critic_loss = 0.5*tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2, axis=0)
            loss = actor_loss + critic_loss


        variables = self.actorcritic_agent.trainable_variables 
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.total_rewards = self.total_rewards + rewards[0]
        
        if self.use_LSTM: 
            obs= original_obs

        random_choice = -1
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


    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()[0]
            if self.use_LSTM: self.eval_memory.clear()
            total_reward = 0
            while(True):
            
                obs = obs.reshape((1,self.obs_shape[0]))

                if self.use_LSTM:
                    obs = self.fill_lstm_buffer(obs, eval= True)
                
                action, _,_,_= self.actorcritic_agent(obs, inference = True)
                if self.environment_name in ["BipedalWalker-v3"]:
                    action = action[0,:]

                obs, reward, terminated, truncated , info = eval_env.step(action)
                reward = reward/self.reward_scaler
                total_reward += reward

                done = truncated or terminated 
                if done:
                    break

            rewards_history.append(total_reward)
            eval_env.close()

        return rewards_history


    def evaluate_agent(self, epoch, sucess_criteria_epochs = 100):
        if (epoch %  self.evaluation_epoch == 0 )  and epoch >0:
            rewards_history = np.mean(self.evaluate(self.env , n_tries=1))
            self.rewards_val_history.append(rewards_history)

            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', self.rewards_val_history[-1], step=epoch)

                if len(self.rewards_val_history)> sucess_criteria_epochs: 
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-sucess_criteria_epochs:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-sucess_criteria_epochs:]), step=epoch)


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



class ContinuousActorCriticAgent(tf.keras.Model):
    def __init__(self,  env,  d, lstm_unit,  use_LSTM=False, sigma_noise = 1e-5):
        super(ContinuousActorCriticAgent, self).__init__()
        self.env = env
        self.d0 = Dense(d, activation='relu', name ="inp0")
        self.d_state_value = Dense(1, activation='linear', name ="v")
        self.use_LSTM = use_LSTM
        self.lstm = LSTM(lstm_unit, return_sequences=False, return_state = False)
        self.state_buffer = None
        self.sigma_noise  = sigma_noise

        self.mean_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.tanh, name='actor_mu' )
        self.sigma_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.softplus, name='actor_sigma' )

    def call(self, observations, inference = False):
        
        if self.use_LSTM:
            x = self.lstm(observations)
        else:
            x = observations

        x = self.d0(x)

        mu = self.mean_dense(x)
        sigma = self.sigma_dense(x)
        mu, sigma = tf.squeeze(mu*2), tf.squeeze(sigma + self.sigma_noise)

        normal_dist = tfp.distributions.Normal(mu, sigma)
        action = tf.clip_by_value(normal_dist.sample(1), self.env.action_space.low[0], self.env.action_space.high[0])
        state_value = self.d_state_value(x)
        if inference:
            return action, state_value[:,0], mu, sigma
        return action[0,:], state_value[:,0], mu, sigma
    

class ActorCriticAgent(tf.keras.Model):
    def __init__(self, state_shape, n_actions, nEnviroment, end_of_episode, d, lstm_unit,  use_LSTM=False):
        super(ActorCriticAgent, self).__init__()
        self.n_actions = n_actions
        self.nEnviroment = nEnviroment
        self.d0 = Dense(d, activation='relu', name ="inp0")
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
    discount, time_to_update, dense_units,  lstm_units, n_enviroment, writer,  use_LSTM = False, trial_n = "", end_of_episode = 200, 
    evaluation_epoch = 2500, environment_name="MountainCar-v0" , reward_scaler = 1):
        super(A3CAgent_Optimization, self).__init__()
        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        obs_shape = self.env.observation_space.shape
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
        self.reward_scaler = reward_scaler

        self.actorcritic_agent = ActorCriticAgent(
                    state_shape = obs_shape,
                    n_actions = self.n_actions, 
                    nEnviroment = n_enviroment,
                    end_of_episode =end_of_episode,
                    d = dense_units,
                    lstm_unit = lstm_units,
                    use_LSTM = use_LSTM
                    )
        
        self.train_env =  gym.make(environment_name).env 
        
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

        self.train_envs_list = [gym.make(environment_name).env  for _ in range(n_enviroment)]#
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

    def sample_actions(self, logits, epsilon, boltzman_factor=1,  exploration = "soft", inference = False, print_policy = False):

        if inference:
            policy = tf.stop_gradient(tf.nn.softmax(logits, axis=-1)).numpy()
            return np.array([np.argmax(p.reshape((1,p.shape[0])), axis = 1) for p in policy])[:,0], 4
        
        policy = tf.stop_gradient(tf.nn.softmax(logits/boltzman_factor, axis=-1)).numpy()
        if exploration == "epsilon":
            rand = np.random.random()
            if rand < epsilon:
                return np.random.randint(self.n_actions, size=len(policy), dtype = np.int64), 1
            
            return  np.array([np.random.choice(len(p), p=p) for p in policy]), 2
        if exploration == "boltzman":
            return np.array([np.random.choice(len(p), p=p) for p in policy]), 3 
        
        return np.array([np.random.choice(len(p), p=p) for p in policy]), 0
       
    

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()[0]
            if self.use_LSTM: self.eval_memory.clear()
            total_reward = 0
            while(True):
            
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
            eval_env.close()

        return rewards_history


    def evaluate_agent(self, epoch, sucess_criteria_epochs = 100):
        if (epoch %  self.evaluation_epoch == 0 )  and epoch >0:
            rewards_history = np.mean(self.evaluate(self.env , n_tries=1))
            self.rewards_val_history.append(rewards_history)

            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', self.rewards_val_history[-1], step=epoch)

                if len(self.rewards_val_history)> sucess_criteria_epochs: 
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-sucess_criteria_epochs:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-sucess_criteria_epochs:]), step=epoch)


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
        



class DQNAgent(tf.keras.Model):
    def __init__(self, n_actions, d):
        super(DQNAgent, self).__init__()
        self.n_actions = n_actions
        self.d0 = Dense(d, activation='relu', name ="inp0")
        self.Qvalues = Dense(self.n_actions, activation='linear', name ="logits")

    def call(self, observations):
        
        x = observations
        x = self.d0(x)
        logits = self.Qvalues(x)

        return logits
    

class MyHyperModel(kt.HyperModel):
    def __init__(self, hyper_dir, writer = "logs_hyper/A3C/" , use_LSTM=False, exploration_tech=None,
                  end_of_episode = 1000, n_enviroment = 5, 
                  evaluation_epoch = 2000, training_steps = 600000,
                  sucess_criteria_epochs = 100, sucess_criteria_value= -100,
                  discount_min = 0.96, discount_max = 0.99,
                  entropy_min = 0.005, entropy_max = 0.05,
                  lr_min = 0.0001, lr_max = 0.005,
                  dense_min = 80, dense_max = 200,
                  lstm_min = 32, lstm_max = 128,
                  time_to_update_min = 100, time_to_update_max=400,
                  environment_name="MountainCar-v0" , 
                  sigma_noise_min = 1e-5, sigma_noise_max = 1e-2,
                  continuous_actions_space = False , reward_scaler = 1):
        
        self.use_LSTM = use_LSTM
        self.writer = writer
        self.hyper_dir = hyper_dir 
        self.n_enviroment =  n_enviroment
        self.evaluation_epoch = evaluation_epoch
        self.exploration_tech = exploration_tech
        self.training_steps = training_steps
        self.sucess_criteria_epochs = sucess_criteria_epochs
        self.sucess_criteria_value = sucess_criteria_value

        self.end_of_episode = end_of_episode
        self.discount_min = discount_min
        self.discount_max = discount_max
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.dense_min = dense_min
        self.dense_max = dense_max
        self.lstm_min = lstm_min
        self.lstm_max = lstm_max
        self.time_to_update_min = time_to_update_min
        self.time_to_update_max = time_to_update_max
        self.environment_name  = environment_name
        self.sigma_noise_min = sigma_noise_min
        self.sigma_noise_max = sigma_noise_max

        self.continuous_actions_space = continuous_actions_space
        self.reward_scaler = reward_scaler

    def build(self, hp):

        end_of_episode = self.end_of_episode
        discount = discount = hp.Float('discount',self.discount_max, self.discount_max, step =0.01)
        entropy_factor =  hp.Float('entropy_factor', self.entropy_min, self.entropy_max) 
        lr = hp.Float('learning_rate', self.lr_min, self.lr_max)
        dense_units =  hp.Int('dense_units', self.dense_min, self.dense_max)

        if self.continuous_actions_space:
            sigma_noise= hp.Float('sigma_noise', self.sigma_noise_min, self.sigma_noise_max)

        if self.use_LSTM : 
            lstm_units = hp.Int('lstm_units', min_value=self.lstm_min, max_value=self.lstm_max)
        else:
            lstm_units = 32
        
        if self.exploration_tech is None :    
            exploration_tech = 'soft'
            time_to_update = end_of_episode
            
        else:
            exploration_tech = self.exploration_tech
            time_to_update = hp.Int('time_to_update', min_value=self.time_to_update_min, max_value=self.time_to_update_max, step=100) 

        if self.continuous_actions_space:
                
            actorcritic_agent = ContinuousA3CAgent_Optimization(
                lr = lr, 
                entropy_factor = entropy_factor,
                discount = discount, 
                dense_units= dense_units, 
                lstm_units = lstm_units,
                n_enviroment =  self.n_enviroment,
                writer = self.writer,
                trial_n = get_valid_trials_number(self.hyper_dir ),
                use_LSTM=self.use_LSTM,
                end_of_episode = end_of_episode,
                evaluation_epoch = self.evaluation_epoch,
                environment_name = self.environment_name,
                sigma_noise= sigma_noise,
                reward_scaler = self.reward_scaler
                )
                
        else:
            actorcritic_agent = A3CAgent_Optimization(
                lr = lr, 
                entropy_factor = entropy_factor,
                exploration_tech = exploration_tech,
                discount = discount, 
                time_to_update = time_to_update, 
                dense_units= dense_units, 
                lstm_units = lstm_units,
                n_enviroment =  self.n_enviroment,
                writer = self.writer,
                trial_n = get_valid_trials_number(self.hyper_dir ),
                use_LSTM=self.use_LSTM,
                end_of_episode = end_of_episode,
                evaluation_epoch = self.evaluation_epoch,
                environment_name = self.environment_name,
                reward_scaler = self.reward_scaler
            )
        
        return actorcritic_agent

    def fit(self, hp, model, x, y,  callbacks=None, **kwargs):
        
        training_steps = self.training_steps
        
        # Record the best validation loss value
        best_epoch_loss = float(-100000)
        
        # Assign the model to the callbacks.rewards_val_history
        for callback in callbacks:
            callback.model = model

        obs = model.env_reset()

        for epoch in range(training_steps):
            
            new_obs = model.run_and_train_agent(epoch, obs)
            model.evaluate_agent(epoch)

            obs = new_obs

            if epoch %  model.evaluation_epoch == 0 and len(model.rewards_val_history)> 0 and len(model.rewards_train_history)> 0 and epoch >0:
                print(f"Epoch: {epoch} : Reward eval/Train: {model.rewards_val_history[-1]}/{model.rewards_train_history[-1]} | epsilon : {model.epsilon}")

            if len(model.rewards_val_history)> self.sucess_criteria_epochs :
                if np.mean(model.rewards_val_history[-self.sucess_criteria_epochs:]) >= self.sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break
        
        final_reward = np.mean(model.rewards_train_history[-self.sucess_criteria_epochs:])
        best_epoch_loss = max(best_epoch_loss, final_reward)

        for callback in callbacks:
            # The "my_metric" is the objective passed to the tuner.
            callback.on_epoch_end(epoch, logs={"total_train_reward" : final_reward})#total_eval_reward

        # Return the evaluation metric value.
        return best_epoch_loss



def merge_JsonFiles(main_hyper_dir, logs_dir, filename):
    result = list()
    res_file = logs_dir+"merged_results.json"
    
    if os.path.isfile(res_file):
        with open(res_file, 'r') as f:
            print("Loading existing merged results file...")
            complete_file = json.load(f)
    else :
        print("Creating merged results file...")
        with open(res_file, mode='w', encoding='utf-8') as f:
            json.dump([], f)
        complete_file = []

    for f1 in filename:
        with open(logs_dir+f1, 'r') as infile:
            complete_file.append(json.load(infile))

        shutil.move(logs_dir+f1, logs_dir+"archived"+os.path.sep+f1)

    with open(res_file, 'w') as output_file:
        print("Updating merged results file...")
        json.dump(complete_file, output_file)

def merge_JsonFiles(main_hyper_dir, logs_dir, filename):
    result = list()
    res_file = logs_dir+"merged_results.json"
    
    if os.path.isfile(res_file):
        with open(res_file, 'r') as f:
            print("Loading existing merged results file...")
            complete_file = json.load(f)
    else :
        print("Creating merged results file...")
        with open(res_file, mode='w', encoding='utf-8') as f:
            json.dump([], f)
        complete_file = []

    for f1 in filename:
        with open(logs_dir+f1, 'r') as infile:
            complete_file.append(json.load(infile))

        shutil.move(logs_dir+f1, logs_dir+"archived"+os.path.sep+f1)

    with open(res_file, 'w') as output_file:
        print("Updating merged results file...")
        json.dump(complete_file, output_file)


def run_hyperparam(START_AGAIN = False, TUNING_TYPE= "", logs_dir="",  main_hyper_dir = "", conda_python_exec= "", py_file = "mountaincar_A3C.py" , hyperparam_combination = [],  total_files = 5):

    if TUNING_TYPE == "MANUAL":
        counter_ = []

        if START_AGAIN:
            if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
                shutil.rmtree(logs_dir)

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            os.makedirs(logs_dir+"archived")
        else:
            counter_ = [int(name.split("_")[-1].split(".")[0]) for name in os.listdir(logs_dir+"archived") if os.path.isfile(os.path.join(logs_dir+"archived", name)) and name != "logfile.txt" and name != "merged_results.json"]
            counter_ = list(np.sort(counter_))

        files_cnt = 0

        with open(logs_dir+'logfile.txt', 'w') as f:
                        
            for i, (n_env, disc, end_ep, lr, entropy_fact, ep, bolt_fact, exploration_tech, train_steps, d, time_to_update, ulstm) in enumerate(hyperparam_combination):
   
                if files_cnt < total_files:
                    print(datetime.now().strftime("%H:%M:%S"), n_env, disc, end_ep, lr, entropy_fact, ep, bolt_fact, exploration_tech, train_steps, d, time_to_update, int(ulstm))
                    commands = main_hyper_dir+py_file+' --n_env '+str(n_env)+' --discount '+str(disc)+' --end_of_episode '+str(end_ep)+' --lr '+str(lr)+' --entropy_fact '+str(entropy_fact)+' --ep '+str(ep)+' --bolt_fact '+str(bolt_fact)+' --exploration_tech '+exploration_tech+' --train_steps '+str(train_steps)+' --file_name testfile_'+str(i)+' --dense_units '+str(d)+' --time_to_update '+str(time_to_update)+' --use_LSTM '+str(int(ulstm))

                    subs = subprocess.Popen((conda_python_exec + commands).split(), stdout=f)
                    
                    files_cnt += 1
                    print("Running : ",i,"/",len(hyperparam_combination))

                if files_cnt >= total_files:
                    while True :
                        time.sleep(60)
                        files = [name for name in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, name)) and name != "logfile.txt" and name != "merged_results.json"]
                        if len(files) >= 1: 
                            merge_JsonFiles(main_hyper_dir, logs_dir, files)
                            files_cnt -= (len(files))
                            if files_cnt < total_files:

                                if files_cnt< 0 :
                                    files_cnt = 0
                                break
                    print("--------------------------")
                time.sleep(1)
    else :
        print("MANUAL tunning type is disabled but run_hyperparam() function was called")


def run_training(training_steps, learning_rate, entropy_factor, exploration_tech, discount, time_to_update, dense_units, lstm_units, 
                 n_enviroment, writer, use_LSTM, end_of_episode, save_factor=50000, sucess_criteria_epochs =100, sucess_criteria_value = -100, 
                 environment_name="MountainCar-v0", reward_scaler = 1, continuous_space_actions = False, sigma_noise = 1e-5, return_agent = False):
    
    if continuous_space_actions:
        model = ContinuousA3CAgent_Optimization(
            lr = learning_rate, 
            entropy_factor = entropy_factor,
            discount = discount, 
            dense_units= dense_units, 
            lstm_units = lstm_units,
            n_enviroment = n_enviroment,
            writer = writer,
            trial_n = get_valid_trials_number(writer),
            use_LSTM=use_LSTM,
            end_of_episode = end_of_episode,
            environment_name = environment_name,
            sigma_noise= sigma_noise,
            reward_scaler = reward_scaler
        )

    else:
        model = A3CAgent_Optimization(
            lr = learning_rate, 
            entropy_factor = entropy_factor,
            exploration_tech = exploration_tech,
            discount = discount, 
            time_to_update = time_to_update, 
            dense_units= dense_units, 
            lstm_units = lstm_units,
            n_enviroment = n_enviroment,
            writer = writer,
            trial_n = get_valid_trials_number(writer),
            use_LSTM=use_LSTM,
            end_of_episode = end_of_episode,
            environment_name = environment_name,
            reward_scaler = reward_scaler
        )

    obs = model.env_reset()
    with tqdm.trange(training_steps) as t:
        for epoch in t:
            new_obs = model.run_and_train_agent(epoch, obs)
            model.evaluate_agent(epoch)

            obs = new_obs
            
            if epoch %save_factor == 0: 
                model.save_weights('./checkpoints/A2Cagent')

            if len(model.rewards_val_history)> sucess_criteria_epochs :
                if np.mean(model.rewards_val_history[-sucess_criteria_epochs:]) >= sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break
                
    # del model
    # tf.keras.backend.clear_session()
    if return_agent:
        return model


def final_evaluation(eval_model, eval_env, n_tries=1, exploration ="soft", video_name = "./A3C_soft_video.mp4", sucess_criteria_epochs= 100, reward_scaler = 1, continuous_action_space = False):
    rewards_history = []
    log_dir = "Evaluation_process/A3C_"+str(exploration)+"/" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summary_writer = tf.summary.create_file_writer(log_dir)

    for k in range(n_tries):

        if k == 0 : video = VideoRecorder(eval_env, path=video_name)

        obs = eval_env.reset()[0]
        total_reward = 0

        log_dir_trial = "Evaluation_process/A3C_trial_"+str(exploration)+"/" + str(k)+ datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_summary_writer_trial = tf.summary.create_file_writer(log_dir_trial)
        
        epoch = 0
        while(True):
    
            if k == 0 : video.capture_frame()
            obs = obs.reshape((1,eval_model.obs_shape[0]))
            if eval_model.use_LSTM:
                obs = eval_model.fill_lstm_buffer(obs, eval= True)
            
            if continuous_action_space:
                action, _, _, _= eval_model.actorcritic_agent(obs, inference= True)
            else: 
                logit, _= eval_model.actorcritic_agent(obs, inference= True)
                action,_ = eval_model.sample_actions(logit,0, boltzman_factor=1, inference= True)
                action = action[0]

            obs, reward, terminated, truncated , info = eval_env.step(action)
            reward = reward/reward_scaler

            total_reward += reward

            done = truncated or terminated 

            with tb_summary_writer_trial.as_default():
                tf.summary.scalar('Final_eval_rewards', total_reward, step=int(epoch) )
                tf.summary.scalar('Final_eval_state', obs[0], step=int(epoch) )
                tf.summary.scalar('Final_eval_velocity', obs[1], step=int(epoch) )
                
            if done:
                break

            epoch +=1

        rewards_history.append(total_reward)
        with tb_summary_writer.as_default():
            tf.summary.scalar('Rewards_history', total_reward, step=int(k) )

            if len(rewards_history) >sucess_criteria_epochs:
                tf.summary.scalar('Average_rewards_history', np.mean(rewards_history[-sucess_criteria_epochs:]), step=int(k) )

        eval_env.close()
        if k == 0 : video.close()

        
    return rewards_history

