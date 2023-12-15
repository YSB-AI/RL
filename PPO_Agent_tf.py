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
from tensorflow.keras.losses import MeanSquaredError
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
import tf_agents


import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder
tf.config.run_functions_eagerly(True)

print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)


class PPO(tf.keras.Model):
    def __init__(self,  discount, dense_units_act, dense_units_crit,  num_layer_act, num_layer_crit, writer,  lr_actor, lr_critic, trial_n = "", 
    evaluation_epoch = 2500, environment_name="", reward_scaler  = 1, gae_lambda=0.95, policy_clip = 0.2, training_epoch=20, entropy_coeff = 0.05, memory_size = 50):
        super(PPO, self).__init__()
        
        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.evaluation_epoch = evaluation_epoch
        self.environment_name= environment_name
        self.epsilon = 1
        self.policy_clip = tf.constant(policy_clip, dtype= tf.float32)
        self.memory_size = memory_size
        
        self.memory = deque(maxlen= self.memory_size)
        self.episode_reward = 0 
        self.training_epoch = training_epoch
        self.entropy_coeff = tf.constant(entropy_coeff, dtype= tf.float32)
        self.epoch_counter = 0
        self.episode_counter = 0
        
        # Training parameters

        self.gae_lambda=tf.constant(gae_lambda, dtype= tf.float32)
        self.discount = tf.constant(discount, dtype= tf.float32)
        self.actor_optimizer=Adam(learning_rate=lr_actor)
        self.critic_optimizer=Adam(learning_rate=lr_critic)

        self.reward_scaler = reward_scaler

        self.actor = Actor(env = self.env,  d = dense_units_act, num_layer_act = num_layer_act)
        self.critic = Critic(env = self.env, d = dense_units_crit,  num_layer_crit = num_layer_crit)

        self.train_env =  gym.make(environment_name)
        _ = self.train_env.reset()
        
        # Logs parameters
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('discount', discount, step=0)
            tf.summary.scalar('gae_lambda', gae_lambda, step=0)
            for i in range(num_layer_act):
                tf.summary.scalar('dense_units_act_'+str(i), dense_units_act[i], step=0)

            for i in range(num_layer_crit):
                tf.summary.scalar('dense_units_crit_'+str(i), dense_units_crit[i], step=0)

            tf.summary.text('environment_name', self.environment_name , step=0)
            tf.summary.scalar('lr_actor', lr_actor , step=0)
            tf.summary.scalar('lr_critic', lr_critic , step=0)
            tf.summary.scalar('reward_scaler', reward_scaler , step=0)


        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewars = 0
    
    @tf.function(reduce_retracing=True)
    def call(self, data):
        epoch, obs = data[0],data[1]
        
        epoch = tf.cast(epoch, dtype = tf.int64)
        obs = tf.reshape(obs, (1,self.obs_shape[0]))
        
        action, log_probs = self.actor.get_action(obs)
        state_value = self.critic(obs)
        if self.environment_name == "BipedalWalker-v3":
            action = action[0,]
        
        next_obs, reward, done, info = self.train_env.step(action)
        self.episode_reward += reward

        self.memory.append([obs, action, log_probs, state_value, reward, done])
        self.append_interaction_metrics(action, reward, done, epoch)

        return next_obs, done
    
    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        _, _ = data
        
        
        def get_sample(next_state_value):
            s = []
            a = []
            log_prob = []
            v = []
            r = []
            dones = []
            #[obs, action, log_probs, state_value, reward, done,])
            for each in self.memory: 

                s.append(each[0])
                a.append(each[1])
                log_prob.append(each[2])
                v.append(each[3])
                r.append(each[4])
                dones.append( 1 - int(each[5]))
            
            v.append(next_state_value)
            
            s = tf.reshape(tf.convert_to_tensor(s), (len(self.memory),self.obs_shape[0]))
            
            if self.env.action_space.shape[0] > 1:
                a = tf.convert_to_tensor(a, dtype = tf.float32)
            else:
                a = tf.reshape(tf.convert_to_tensor(a, dtype = tf.float32), (len(a),))

            log_prob =tf.reshape(tf.convert_to_tensor(log_prob, dtype = tf.float32), (len(log_prob), self.env.action_space.shape[0])) 

            v = tf.reshape(tf.convert_to_tensor(v, dtype = tf.float32), (len(v), 1))
            r = tf.reshape(tf.convert_to_tensor(r, dtype = tf.float32), (len(self.memory), 1))
            dones = tf.reshape(tf.convert_to_tensor(dones, dtype = tf.float32), (len(self.memory), 1))

            return s,a,log_prob, v,r,dones
    
        def process_training(obs, action_arr, returns, advantages, old_log_probs):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                
                # Update actor network
                mu, sigma = self.actor(obs, training= True)
                normal_dist = tfp.distributions.Normal(mu, sigma)

                log_probs = tf.cast(normal_dist.log_prob(action_arr), dtype = tf.float32)
                log_probs = tf.reshape(log_probs, (len(log_probs),self.env.action_space.shape[0]))
                
                old_log_probs = tf.reshape(old_log_probs, (len(old_log_probs),self.env.action_space.shape[0]))

                state_value = self.critic(obs, training=True)
                state_value = tf.reshape(state_value, (len(state_value),1))

                prob_ratio = tf.exp(log_probs - old_log_probs) 
                

                weighted_clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                
                # Entropy regularization
                
                entropy = tf.reduce_mean(normal_dist.entropy())
                
                # Update critic network
                critic_loss =tf.reduce_mean((state_value - returns)**2, axis=0) *0.5

                #actor_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages, weighted_clipped_probs * advantages)) + (self.entropy_coeff *  entropy)) + critic_loss
                actor_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages, weighted_clipped_probs * advantages)) + (self.entropy_coeff *  entropy) + critic_loss)

            actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            return actor_loss, critic_loss, state_value, entropy, prob_ratio
            
        
        def train( next_state_value):
            
            obs, action_arr, old_log_probs, state_value_arr, rewards, dones_arr =  get_sample(next_state_value)
            
            final_value = state_value_arr[-1]
            old_state_value = tf.reshape(state_value_arr[:-1], (len(state_value_arr[:-1]),1))

            discounts = self.discount *  dones_arr
            discounts = tf.cast(tf.reshape(discounts, (len(discounts),1)), dtype = tf.float32)

            
            advantages = tf_agents.utils.value_ops.generalized_advantage_estimation(
                values = old_state_value, final_value = final_value, discounts =  discounts, 
                rewards = rewards, td_lambda=self.gae_lambda, time_major=True
            )
            

            returns=  tf_agents.utils.value_ops.discounted_return(rewards,discounts,time_major=True, final_value=old_state_value, provide_all_returns = False) #tf.cast(tf.convert_to_tensor(returns), dtype = tf.float32)
            advantages= tf.cast(tf.convert_to_tensor(advantages), dtype = tf.float32)

            for _ in tf.range(self.training_epoch):
                actor_loss, critic_loss, state_value, entropy, prob_ratio = process_training(obs, action_arr, returns, advantages, old_log_probs)
        


            actor_loss = tf.squeeze(actor_loss, axis = 0)
            critic_loss = tf.squeeze(critic_loss, axis = 0)


            return actor_loss, critic_loss, advantages, state_value, returns, entropy, prob_ratio
        
        #obs = tf.reshape(self.train_env.reset(), (1,self.obs_shape[0]))
        obs = self.train_env.reset()
        self.episode_reward = 0
        
        for _ in tf.range(self.train_env._max_episode_steps):
            next_obs, done = self([self.epoch_counter, tf.reshape(obs, (1,self.obs_shape[0]))], training = True)
            
            if done or len(self.memory)== self.memory_size:        
                next_state_value = self.critic(tf.reshape(next_obs, (1,self.obs_shape[0])))
                actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio = train(next_state_value)
                self.append_training_metrics(self.episode_reward, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, self.episode_counter)
                self.rewards_train_history.append(self.episode_reward)
                
                self.memory.clear()

                self.episode_counter +=1
                break
            
            self.epoch_counter +=1
            
        return {"actor_loss": actor_loss, "total_train_reward": self.episode_reward}
           

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()
            total_reward = 0
            while(True):
                
                
                actions,_ = self.actor.get_action(obs.reshape((1,self.obs_shape[0])))
                if self.environment_name == "BipedalWalker-v3":
                    actions = actions[0,]

                obs, reward, done, info = eval_env.step(actions)
                reward = reward
                total_reward += reward
                #done = truncated or terminated  #terminated, truncated,
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
    


    def append_interaction_metrics(self, action, reward,  done, epoch):
        with self.tb_summary_writer.as_default():
            tf.summary.histogram('Action', action, step=epoch)
            tf.summary.scalar('RT_Reward', reward, step=epoch)
            tf.summary.scalar('Done', done, step=epoch)

    def append_training_metrics(self, episode_reward, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, epoch):

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('Episode_Reward', episode_reward, step=epoch)
            tf.summary.scalar('Entropy', entropy, step=epoch)
            tf.summary.scalar('actor_loss', actor_loss, step=epoch)
            tf.summary.scalar('critic_loss', critic_loss, step=epoch)
            tf.summary.scalar('Advantages', tf.reduce_mean((advantages)), step=epoch)
            tf.summary.scalar('State_Value', tf.reduce_mean(tf.convert_to_tensor(state_values)), step=epoch)
            tf.summary.scalar('Expected_Return', tf.reduce_mean(tf.convert_to_tensor(returns)), step=epoch)
            tf.summary.scalar('Prob_ratio', tf.reduce_mean(tf.convert_to_tensor(prob_ratio)), step=epoch)
            tf.summary.scalar('RT_epsilon', self.epsilon, step=epoch)


class Actor(tf.keras.Model):
    def __init__(self,  env,  d, num_layer_act = 1):
        super(Actor, self).__init__()
        self.env = env
        self.num_layers = num_layer_act
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_act)]
        self.mean_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.tanh, name='actor_mu' )
        self.sigma_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.softplus, name='actor_sigma' ) 

        self.noise= 1e-5

    @tf.function
    def call(self, observations):

        # Get mean and standard deviation from the policy network
        x = observations
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)

        mu = self.mean_dense(x)
        sigma = self.sigma_dense(x)
        mu, sigma = tf.squeeze(mu*self.env.action_space.high[0]), tf.squeeze(sigma + self.noise)

        return mu, sigma
    
    @tf.function
    def get_action(self, observations):
        
        mu, sigma = self.call(observations)
        normal_dist = tfp.distributions.Normal(mu, sigma)

        action = tf.clip_by_value(normal_dist.sample(1) , self.env.action_space.low[0], self.env.action_space.high[0])
        
        log_probs =  normal_dist.log_prob(action)
        
        return action, log_probs



class Critic(tf.keras.Model):
    def __init__(self,  env,  d,  num_layer_crit = 1):
        super(Critic, self).__init__()
        self.env = env
      
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_crit)]
        self.num_layers = num_layer_crit
        self.d_state_value = Dense(1,  name ="v")
    
    @tf.function
    def call(self, observations):
        x = observations
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
        state_value = self.d_state_value(x)
        return state_value[:,0]
    

def get_valid_trials_number(hyper_dir):
    
    dir = "./"+hyper_dir
    trial_n = 0 
    if os.path.isdir(dir) :
        trial_n = len(next(os.walk(dir))[1])
        print("Trial number : ",trial_n)
        
    return str(trial_n)+"_"




def get_model(discount,  dense_units_act,  dense_units_crit,num_layer_a,num_layer_c, writer,  
                 environment_name="MountainCar-v0", reward_scaler = 1, evaluation_epoch = 2000,
                 lr_actor= 0.001, lr_critic= 0.001,   gae_lambda=0.95, training_epoch= 20, entropy_coeff= 0.01, policy_clip = 0.2,memory_size= 50):
  
    model = PPO(
            discount = discount, 
            dense_units_act = dense_units_act,
            dense_units_crit= dense_units_crit, 
            num_layer_act  = num_layer_a, 
            num_layer_crit= num_layer_c,
            writer = writer,
            trial_n = get_valid_trials_number(writer ),
            evaluation_epoch = evaluation_epoch,
            environment_name = environment_name,
            reward_scaler = reward_scaler,
            lr_actor = lr_actor, lr_critic = lr_critic, 
            gae_lambda=gae_lambda,
            policy_clip = policy_clip,
            training_epoch = training_epoch ,
            entropy_coeff = entropy_coeff,
            memory_size = memory_size
            )
    

    #if return_agent:
    return model

def final_evaluation(eval_model, eval_env, n_tries=1, exploration ="soft", video_name = "./PPO_soft_video.mp4", sucess_criteria_epochs= 100, reward_scaler = 1):
    rewards_history = []
    log_dir = "Evaluation_process/A3C_"+str(exploration)+"/" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summary_writer = tf.summary.create_file_writer(log_dir)

    for k in range(n_tries):

        if k == 0 : video = VideoRecorder(eval_env, path=video_name)

        obs = eval_env.reset()
        total_reward = 0

        log_dir_trial = "Evaluation_process/A3C_trial_"+str(exploration)+"/" + str(k)+ datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_summary_writer_trial = tf.summary.create_file_writer(log_dir_trial)
        
        epoch = 0
        while(True):
    
            if k == 0 : video.capture_frame()
            actions, log_probs  = eval_model.actor.get_action(obs.reshape((1,eval_model.obs_shape[0])))
            
            obs, reward, done , info = eval_env.step(actions)
            total_reward += reward
            #done = truncated or terminated  #terminated, truncated 

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

    return np.mean(rewards_history)
