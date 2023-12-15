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
from tensorflow.keras.layers import Dense
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


import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder


print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)



def get_valid_trials_number(hyper_dir):
    
    dir = "./"+hyper_dir
    trial_n = 0 
    if os.path.isdir(dir) :
        trial_n = len(next(os.walk(dir))[1])
        print("Trial number : ",trial_n)
        
    return str(trial_n)+"_"

class SARSA_Agent_Optimization(tf.keras.Model):
    def __init__(self, lr, discount, time_to_update, dense_units, writer, trial_n, end_of_episode =1000, evaluation_epoch = 1000,  batch_size = 32, exploration_tech = "boltzman" , environment_name = "MountainCar-v0"):
        super(SARSA_Agent_Optimization, self).__init__()

        # Enviroment parameters
        self.environment_name = environment_name
        self.env = gym.make(environment_name)
        self.train_env = gym.make(environment_name).env
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n 
        self.use_scale = False
        self.episode_steps = 0 
        self.evaluation_epoch = evaluation_epoch

        # Training parameters
        self.batch_size = batch_size
        self.end_of_episode = end_of_episode
        self.memory = deque(maxlen= 10000)

        self.writer = writer
        self.learning_rate= lr
        self.discount = discount
        self.time_to_update = time_to_update
        self.epsilon = 1
        self.boltzman_factor = 1
        self.optimizer=Adam(learning_rate=lr)
        self.exploration_tech = exploration_tech
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.SARSA_agent = SARSAAgent(
                    n_actions = n_actions, 
                    d = dense_units,
                    )

        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('learning_rate', self.learning_rate, step=0)
            tf.summary.scalar('discount', self.discount, step=0)
            tf.summary.scalar('dense_units', dense_units, step=0)
            tf.summary.scalar('time_to_update', time_to_update, step=0)
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
            while(True):

                current_qvalue = self.SARSA_agent(state.reshape((1,len(state))))
                action,_ = self.sample_actions(current_qvalue,0, 1, exploration="soft", inference= True)
                action = action[0]
                state, reward, terminated, truncated , info = eval_env.step(action)
                total_reward += reward

                done = truncated or terminated 
                if done:
                    break

            rewards_history.append(total_reward)
            eval_env.close()
        
        return rewards_history
    
    def get_sample(self):
        batch = np.array(list(self.memory))

        s = np.array([each[0] for each in batch])
        a = [each[1] for each in batch]
        s_ = np.array([each[2] for each in batch])
        r = [each[3] for each in batch]
        dones = [(1-each[4]) for each in batch]
        a_ = [each[3] for each in batch]
        

        return s,a,r,s_,dones, a_


    def run_agent_and_train(self, epoch, obs):

        self.episode_steps +=1 

        with tf.GradientTape() as tape:
            current_qvalue = self.SARSA_agent(obs.reshape((1,len(obs))))
            actions, random_selection = self.sample_actions(current_qvalue,  self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
            actions = actions[0]
            new_obs, rewards, terminated, truncated , infos= self.train_env.step(actions)
            self.total_rewards = self.total_rewards  + rewards

            done = terminated or truncated

            next_qvalue = self.SARSA_agent(new_obs.reshape((1,len(new_obs))))
            next_actions, next_random_selection = self.sample_actions(next_qvalue,  self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
            next_actions = next_actions[0]

            current_action_qvalues = tf.reduce_sum(tf.one_hot(actions, self.n_actions) * current_qvalue, axis=1)
            next_action_qvalues = tf.reduce_sum(tf.one_hot(next_actions, self.n_actions) * next_qvalue, axis=1)
            
            target_state_values = rewards + tf.math.multiply(self.discount,(done)*next_action_qvalues)
            loss = self.loss_fn(tf.stop_gradient(target_state_values), current_action_qvalues)

        variables = self.SARSA_agent.trainable_variables 
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.append_training_metrics(obs.reshape((1,len(obs))), loss, target_state_values,next_qvalue, current_qvalue, epoch, obs, actions, done, random_selection)
        
        if self.episode_steps %  self.end_of_episode == 0 and epoch >0: 
            
            done = True

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
        
    
    # def run_agent_and_train(self, epoch, obs):

    #     self.episode_steps +=1 
        
    #     current_qvalue = self.SARSA_agent(obs.reshape((1,len(obs))))
    #     actions, random_selection = self.sample_actions(current_qvalue,  self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
    #     actions = actions[0]
    #     new_obs, rewards, terminated, truncated , infos= self.train_env.step(actions)
    #     self.total_rewards = self.total_rewards  + rewards
    #     done = terminated or truncated

    #     next_qvalue = self.SARSA_agent(new_obs.reshape((1,len(new_obs))))
    #     next_actions, next_random_selection = self.sample_actions(next_qvalue,  self.epsilon, self.boltzman_factor, exploration=self.exploration_tech, inference = False)
    #     next_actions = next_actions[0]
    #     self.memory.append([obs, actions, new_obs, rewards, done, next_actions])
        
    #     if len(self.memory) == self.batch_size:

    #         obs_batch, action_batch, reward_batch, new_obs_batch, done_batch, next_action_batch = self.get_sample()

    #         with tf.GradientTape() as tape:
    #             current_qvalue = self.SARSA_agent(obs_batch)
    #             next_qvalue = self.SARSA_agent(new_obs_batch)

    #             current_qvalue = tf.cast(current_qvalue, dtype= tf.float32)
    #             next_qvalue = tf.cast(next_qvalue, dtype= tf.float32)
    #             done_batch = tf.cast(done_batch, dtype= tf.float32)
    #             reward_batch = tf.cast(reward_batch, dtype= tf.float32)
    #             action_batch = tf.cast(action_batch, dtype= tf.int32)
    #             next_action_batch = tf.cast(next_action_batch, dtype= tf.int32)

    #             current_action_qvalues = tf.reduce_sum(tf.one_hot(action_batch, self.n_actions) * current_qvalue, axis=1)
    #             next_action_qvalues = tf.reduce_sum(tf.one_hot(next_action_batch, self.n_actions) * next_qvalue, axis=1)
                
    #             target_state_values = reward_batch + tf.math.multiply(self.discount,(done_batch)*next_action_qvalues)

    #             loss = self.loss_fn(target_state_values, current_action_qvalues)

    #         variables = self.SARSA_agent.trainable_variables 
    #         gradients = tape.gradient(loss, variables)
    #         self.optimizer.apply_gradients(zip(gradients, variables))

    #         self.append_training_metrics(obs.reshape((1,len(obs))), loss, target_state_values,next_qvalue,current_qvalue, epoch, obs, actions, done, random_selection)
    #         # self.memory.clear()
            
    #     if self.episode_steps %  self.end_of_episode == 0 and epoch >0: done = True

    #     obs = new_obs
    #     if done and epoch>0 : 
    #         self.rewards_train_history.append(self.total_rewards)
            
    #         with self.tb_summary_writer.as_default():
    #             tf.summary.scalar('Training_rewards', self.total_rewards, step= len(self.rewards_train_history) )
    #             tf.summary.scalar('RT_done', 0, step=epoch)
    #             tf.summary.scalar('RT_done', 1, step=epoch+1)

    #         self.total_rewards = 0
    #         obs = self.train_env.reset()[0]  
    #         self.episode_steps = 0 

    #     return obs
        

    def update_training_params(self, epoch):

        if epoch% self.time_to_update == 0 and self.exploration_tech == "epsilon" and self.epsilon > 0.0001 :
            self.epsilon = self.epsilon*0.99
            
        elif epoch% self.time_to_update == 0 and self.exploration_tech == "boltzman" and self.boltzman_factor > 0.01 :
            self.boltzman_factor = self.boltzman_factor*0.99
        

    def evaluate_agent(self, epoch, sucess_criteria_epochs= 100):

        if epoch %self.evaluation_epoch ==0 and epoch >0:
            rewards_history = np.mean(self.evaluate(self.env , n_tries=3))
            self.rewards_val_history.append(rewards_history)

            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', rewards_history, step=int(epoch/self.end_of_episode) )
                
                if len(self.rewards_val_history)> sucess_criteria_epochs: 
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-sucess_criteria_epochs:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-sucess_criteria_epochs:]), step=epoch)


    def append_training_metrics(self,  _batch_states,  loss,_target_state_values, next_qvalue_batch, current_qvalue_batch, epoch,obs, actions, done, random_selection):

        if epoch %50 == 0:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Loss', loss, step=epoch)
                tf.summary.scalar('Target_history', tf.reduce_mean(tf.convert_to_tensor(_target_state_values), axis = 0), step=epoch)
                tf.summary.scalar("Training_state", _batch_states[0][0], step=epoch)
                tf.summary.scalar("Training_velocity", _batch_states[0][1], step=epoch)
                tf.summary.scalar('RT_action', actions, step=epoch)
                tf.summary.scalar("RT_state", obs[0], step=epoch)
                tf.summary.scalar("RT_velocity", obs[1], step=epoch)
                tf.summary.scalar("RT_epsilon", self.epsilon, step=epoch)
                tf.summary.scalar('RT_boltzman_factor', self.boltzman_factor, step=epoch)
                tf.summary.scalar('RT_rewards', self.total_rewards, step=epoch)
                tf.summary.scalar('random_selection', random_selection, step=epoch)


class SARSAAgent(tf.keras.Model):
    def __init__(self, n_actions, d):
        super(SARSAAgent, self).__init__()
        self.n_actions = n_actions
        self.d0 = Dense(d, activation='relu', name ="inp0")
        self.Qvalues = Dense(self.n_actions, activation='linear', name ="logits")

    def call(self, observations):
        
        x = observations
        x = self.d0(x)
        logits = self.Qvalues(x)

        return logits
    

class MyHyperModel(kt.HyperModel):
    def __init__(self, hyper_dir,  writer = "logs_SARSA_V3/hyper/", exploration_tech ='epsilon',
                 sucess_criteria_epochs = 100, sucess_criteria_value = -100,
                  end_of_episode = 1000, batch = 256,  evaluation_epoch = 2000, 
                   training_steps = 2000000, 
                   time_to_update_min = 400, time_to_update_max = 800,
                   lr_min = 0.00005, lr_max = 0.005,
                   discount_min = 0.97, discount_max = 0.99,
                   dense_min = 100, dense_max = 160, environment_name="MountainCar-v0" ):
                   
        self.writer = writer
        self.hyper_dir = hyper_dir  
        self.writer = writer
        self.exploration_tech =exploration_tech
        self.sucess_criteria_epochs = sucess_criteria_epochs
        self.sucess_criteria_value = sucess_criteria_value
        self.end_of_episode = end_of_episode
        self.batch = batch
        self.evaluation_epoch = evaluation_epoch
        self.training_steps = training_steps
        self.discount_min = discount_min
        self.discount_max = discount_max
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.time_to_update_min = time_to_update_min
        self.time_to_update_max = time_to_update_max
        self.dense_min = dense_min
        self.dense_max = dense_max
        self.environment_name = environment_name

    def build(self, hp):
        end_of_episode = self.end_of_episode
        discount = hp.Float('discount', self.discount_min, self.discount_max, step =0.01)
        lr = hp.Float('learning_rate', self.lr_min, self.lr_max )
        time_to_update =  hp.Int('time_to_update', min_value= self.time_to_update_min, max_value=self.time_to_update_max, step=100)
        batch = self.batch
        dense_units = hp.Int('dense_units',  min_value=self.dense_min, max_value=self.dense_max)
        exploration_tech = self.exploration_tech

        agent = SARSA_Agent_Optimization(
            lr = lr, 
            discount = discount, 
            time_to_update = time_to_update, 
            dense_units= dense_units, 
            writer = self.writer ,
            end_of_episode = end_of_episode,
            exploration_tech = exploration_tech,
            trial_n = get_valid_trials_number(self.hyper_dir),
            evaluation_epoch = self.evaluation_epoch,
            batch_size = batch,
            environment_name=self.environment_name)
        
        return agent

    def fit(self, hp, model, x, y,  callbacks=None, **kwargs):
        training_steps = self.training_steps
        
        best_epoch_loss = float(-100000)

        for callback in callbacks:
            callback.model = model

        obs = model.train_env.reset()[0]
        
        for epoch in range(training_steps):

            obs = model.run_agent_and_train(epoch, obs)
            model.update_training_params(epoch)
            model.evaluate_agent(epoch)

            if epoch %  model.evaluation_epoch == 0 and len(model.rewards_val_history)> 0 and len(model.rewards_train_history)> 0 and epoch >0:
                print(f"Epoch: {epoch} : Reward eval/Train: {model.rewards_val_history[-1]}/{model.rewards_train_history[-1]} | epsilon : {model.epsilon}| boltzman : {model.boltzman_factor}")

            if len(model.rewards_val_history)> self.sucess_criteria_epochs :
                if np.mean(model.rewards_val_history[-self.sucess_criteria_epochs:]) >= self.sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break

        final_reward = np.mean(model.rewards_train_history[-self.sucess_criteria_epochs:])
        best_epoch_loss = max(best_epoch_loss, final_reward)

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={"total_train_reward" : final_reward})

        # Return the evaluation metric value.
        return best_epoch_loss


def run_training(training_steps, learning_rate, exploration_tech, discount, time_to_update, dense_units, writer, end_of_episode, sucess_criteria_epochs = 100, sucess_criteria_value = -100):

    model = SARSA_Agent_Optimization(
        lr = learning_rate, 
        discount = discount, 
        time_to_update = time_to_update, 
        dense_units= dense_units, 
        writer = writer ,
        end_of_episode = end_of_episode,
        exploration_tech = exploration_tech,
        trial_n = get_valid_trials_number(writer),
        evaluation_epoch = 2000)


    obs = model.train_env.reset()[0]

    with tqdm.trange(0, training_steps) as t:
        for epoch in t:

            obs = model.run_agent_and_train(epoch, obs)
            model.update_training_params(epoch)
            model.evaluate_agent(epoch)
            if epoch %5000 == 0: 
                model.save_weights('./checkpoints/SARSAagent')
            
            if len(model.rewards_val_history)> sucess_criteria_epochs :
                if np.mean(model.rewards_val_history[-sucess_criteria_epochs:]) >= sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break
    
    del model
    tf.keras.backend.clear_session()

def final_evaluation(eval_model, eval_env, n_tries=1, exploration ="epsilon",  video_name = "./SARSA_epsilon_video.mp4", sucess_criteria_epochs = 100):

    rewards_history = []
    log_dir = "Evaluation_process/SARSA_"+str(exploration)+"/" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summary_writer = tf.summary.create_file_writer(log_dir)

    for k in range(n_tries):

        if k == 0: video = VideoRecorder(eval_env, path=video_name)

        state = eval_env.reset()[0]
        total_reward = 0

        log_dir_trial = "Evaluation_process/SARSA_trial_"+str(exploration)+"/" + str(k)+ datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_summary_writer_trial = tf.summary.create_file_writer(log_dir_trial)

        epoch = 0
        while(True):
    
            if k == 0: video.capture_frame()
            current_qvalue = eval_model.SARSA_agent(state.reshape((1,len(state))))
            action,_ = eval_model.sample_actions(current_qvalue,0, 1, exploration="soft", inference= True)
            action = action[0]
            state, reward, terminated, truncated , info = eval_env.step(action)
            total_reward += reward

            done = truncated or terminated 
            with tb_summary_writer_trial.as_default():
                tf.summary.scalar('Final_eval_rewards', total_reward, step=int(epoch) )
                tf.summary.scalar('Final_eval_state', state[0], step=int(epoch) )
                tf.summary.scalar('Final_eval_velocity', state[1], step=int(epoch) )

            if done:
                break

            epoch +=1
            
        rewards_history.append(total_reward)
        with tb_summary_writer.as_default():
            tf.summary.scalar('Rewards_history', total_reward, step=int(k) )

            if len(rewards_history) >sucess_criteria_epochs:
                tf.summary.scalar('Average_rewards_history', np.mean(rewards_history[-sucess_criteria_epochs:]), step=int(k) )

        eval_env.close()
        if k == 0: video.close()

        
    return rewards_history

