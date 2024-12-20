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


print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)


class PPO(tf.keras.Model):
    def __init__(self,  discount, dense_units, num_layer, writer,  learning_rate, trial_n = "", 
    evaluation_epoch = 2500, environment_name="", reward_scaler  = 1, gae_lambda=0.95, policy_clip = 0.2, training_epoch=20, entropy_coeff = 0.05, memory_size = 50):
        super(PPO, self).__init__()
        
        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.evaluation_epoch = evaluation_epoch
        self.environment_name= environment_name
        self.epsilon = 1
        self.policy_clip = policy_clip
        self.memory_size = memory_size
        
        self.memory = deque(maxlen= self.memory_size)
        self.episode_reward = 0 
        self.training_epoch = training_epoch
        self.entropy_coeff = entropy_coeff
        
        # Training parameters

        self.gae_lambda=gae_lambda
        self.discount = discount
        self.optimizer=Adam(learning_rate=learning_rate)

        self.reward_scaler = reward_scaler

        self.actor_critic = Actor_Critic(env = self.env,  d = dense_units, num_layer = num_layer)

        self.train_env =  gym.make(environment_name)
        
        # Logs parameters
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('discount', discount, step=0)
            tf.summary.scalar('gae_lambda', gae_lambda, step=0)
            for i in range(num_layer):
                tf.summary.scalar('dense_units'+str(i), dense_units[i], step=0)

            tf.summary.text('environment_name', self.environment_name , step=0)
            tf.summary.scalar('learning_rate', learning_rate , step=0)
            tf.summary.scalar('reward_scaler', reward_scaler , step=0)


        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewars = 0
      
    def compute_advantages(self, rewards, values, dones):
        g = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.discount * values[i + 1] * dones[i] - values[i]
            g = delta + self.discount * self.gae_lambda * dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32)
        adv = adv.reshape((len(adv),))
        adv = adv - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        returns = np.array(returns, dtype=np.float32).reshape((len(returns),))
        adv = adv.reshape((len(adv),))
        
        return returns, adv 


    def get_sample(self,next_state_value):
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
        
        s = np.array(s).reshape((len(self.memory), self.obs_shape[0]))
        
        if self.env.action_space.shape[0] > 1:
            a = np.array(a)
        else:
            a = np.array(a).reshape((len(a)))

        log_prob = np.array(log_prob)

        v = np.array(v).reshape((len(v), 1))
        r = np.array(r).reshape((len(self.memory), 1))
        dones = np.array(dones).reshape((len(self.memory), 1))

        return s,a,log_prob, v,r,dones
    
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    
    def process_training(self,obs, action_arr, returns, advantages, old_log_probs):
        with tf.GradientTape(persistent = True) as tape:
            
            # Update actor network
            mu, sigma, state_value = self.actor_critic(obs)
            normal_dist = tfp.distributions.Normal(mu, sigma)

            log_probs = tf.cast(normal_dist.log_prob(action_arr), dtype = tf.float32)
            log_probs = tf.reshape(log_probs, (len(log_probs),self.env.action_space.shape[0]))
            state_value = tf.reshape(state_value, (len(state_value),1))
            
            old_log_probs = tf.reshape(old_log_probs, (len(old_log_probs),self.env.action_space.shape[0]))
            prob_ratio = tf.exp(log_probs - old_log_probs) 
            
            weighted_clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
            
            # Entropy regularization
            entropy = tf.reduce_mean(normal_dist.entropy())
            
            # Update critic network
            critic_loss =tf.reduce_mean((state_value - returns)**2, axis=0) *0.5
            critic_loss = tf.squeeze(critic_loss, axis = 0)
            actor_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages, weighted_clipped_probs * advantages)) + (self.entropy_coeff *  entropy) )
            
            total_loss = actor_loss + critic_loss

        
        actor_critic_grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_critic_grads, self.actor_critic.trainable_variables))
        
        return total_loss, actor_loss, critic_loss, state_value, entropy, prob_ratio
        
        
    def train(self, next_state_value):
        
        state_arr, action_arr, old_prob_arr, state_value_arr, reward_arr, dones_arr =  self.get_sample(next_state_value)
        
        rewards = tf.cast(tf.reshape(reward_arr, (len(reward_arr),1)), dtype = tf.float32)
        final_value = tf.cast(state_value_arr[-1], dtype = tf.float32)
        old_state_value = tf.cast(tf.reshape(state_value_arr[:-1], (len(state_value_arr[:-1]),1)), dtype = tf.float32)

        discounts = self.discount *  dones_arr
        discounts = tf.cast(np.array(discounts).reshape((len(discounts),1)), dtype = tf.float32)

        
        advantages = tf_agents.utils.value_ops.generalized_advantage_estimation(
            values = old_state_value, final_value = final_value, discounts =  discounts, 
            rewards = rewards, td_lambda=self.gae_lambda, time_major=True
        )
        

        obs = tf.cast(tf.convert_to_tensor(state_arr), dtype = tf.float32)
        old_log_probs = tf.cast(tf.reshape(tf.convert_to_tensor(old_prob_arr), (len(old_prob_arr), self.env.action_space.shape[0])), dtype = tf.float32)

        returns=  tf_agents.utils.value_ops.discounted_return(rewards,discounts,time_major=True, final_value=old_state_value, provide_all_returns = False) #tf.cast(tf.convert_to_tensor(returns), dtype = tf.float32)
        advantages= tf.cast(tf.convert_to_tensor(advantages), dtype = tf.float32)

        for k in range(self.training_epoch):  
            
            total_loss, actor_loss, critic_loss, state_value, entropy, prob_ratio = self.process_training(obs, action_arr, returns, advantages, old_log_probs)


        # print("discounts->",discounts.shape)
        # print("advantages ->", advantages.shape)
        # print("returns ->", returns.shape)
        # print("state_value ->", state_value.shape)
        # print("log_probs ->", log_probs.shape)
        # print("old_log_probs ->", old_log_probs.shape)
        # print("prob_ratio ->", prob_ratio.shape)
        # print("weighted_clipped_probs ->", weighted_clipped_probs.shape)
        # print("entropy ->", entropy.shape)
        # print("actor_loss ->", actor_loss.shape)
        # print("critic_loss ->", critic_loss.shape)


        return total_loss, actor_loss, critic_loss, advantages, state_value, returns, entropy, prob_ratio

    def run_agent(self, epoch, obs):
        action, log_probs, state_value = self.actor_critic.get_action(obs.reshape((1,self.obs_shape[0])))

        if self.environment_name == "BipedalWalker-v3":
            action = action[0,]
        next_obs, reward, done, info = self.train_env.step(action)
        #done = truncated or terminated  #terminated, truncated,
        self.episode_reward += reward

        
        self.memory.append([obs, action, log_probs, state_value, reward, done])
        self.append_interaction_metrics(action, reward, done, epoch)

        if done or len(self.memory)== self.memory_size:        
            _,_,next_state_value = self.actor_critic(next_obs.reshape((1,self.obs_shape[0])))
            
            total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio = self.train(next_state_value)
            self.append_training_metrics(self.episode_reward, total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, epoch)
            self.rewards_train_history.append(self.episode_reward)

            self.memory.clear()
            self.episode_reward = 0 
            
            if done :
                next_obs = self.train_env.reset()

        return next_obs
        

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()
            total_reward = 0
            while(True):
                
                
                actions,_,_ = self.actor_critic.get_action(obs.reshape((1,self.obs_shape[0])))
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

    def append_training_metrics(self, episode_reward, total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, epoch):

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('Episode_Reward', episode_reward, step=epoch)
            tf.summary.scalar('Entropy', entropy, step=epoch)
            tf.summary.scalar('total_loss', total_loss, step=epoch)
            tf.summary.scalar('actor_loss', actor_loss, step=epoch)
            tf.summary.scalar('critic_loss', critic_loss, step=epoch)
            tf.summary.scalar('Advantages', tf.reduce_mean((advantages)), step=epoch)
            tf.summary.scalar('State_Value', tf.reduce_mean(tf.convert_to_tensor(state_values)), step=epoch)
            tf.summary.scalar('Expected_Return', tf.reduce_mean(tf.convert_to_tensor(returns)), step=epoch)
            tf.summary.scalar('Prob_ratio', tf.reduce_mean(tf.convert_to_tensor(prob_ratio)), step=epoch)
            tf.summary.scalar('RT_epsilon', self.epsilon, step=epoch)


class Actor_Critic(tf.keras.Model):
    def __init__(self,  env,  d, num_layer = 1):
        super(Actor_Critic, self).__init__()
        self.env = env
        self.num_layers = num_layer
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer)]
        self.mean_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.tanh, name='actor_mu' )
        self.sigma_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.softplus, name='actor_sigma' ) 
        self.d_state_value = Dense(1,  name ="v")

        self.noise= 1e-5

    def call(self, observations):

        # Get mean and standard deviation from the policy network
        x = observations
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)

        mu = self.mean_dense(x)
        sigma = self.sigma_dense(x)
        state_value = self.d_state_value(x)
        
        mu, sigma = tf.squeeze(mu*self.env.action_space.high[0]), tf.squeeze(sigma + self.noise)

        return mu, sigma, state_value[:,0]
    
    
    def get_action(self, observations):
        
        mu, sigma,state = self.call(observations)
        normal_dist = tfp.distributions.Normal(mu, sigma)

        action = tf.clip_by_value(normal_dist.sample(1) , self.env.action_space.low[0], self.env.action_space.high[0])
        
        log_probs =  normal_dist.log_prob(action)
        
        return action, log_probs, state
    

def get_valid_trials_number(hyper_dir):
    
    dir = "./"+hyper_dir
    trial_n = 0 
    if os.path.isdir(dir) :
        trial_n = len(next(os.walk(dir))[1])
        print("Trial number : ",trial_n)
        
    return str(trial_n)+"_"


class MyHyperModel(kt.HyperModel):
    def __init__(self, hyper_dir, writer = "logs_hyper/PPO/" , exploration_tech=None,
                  evaluation_epoch = 2000, training_steps = 600000,
                  sucess_criteria_epochs = 100, sucess_criteria_value= -100,
                  discount_min = 0.96, discount_max = 0.99,
                  gae_min = 0.90, gae_max = 0.99,
                  lr_actor_min = 0.00001, lr_actor_max = 0.005,
                  lr_critic_min = 0.00001, lr_critic_max = 0.005,
                  dense_min = 32, dense_max = 200,
                  environment_name="MountainCar-v0" , 
                  reward_scaler = 1, 
                  num_layers_act = 1, num_layers_crit = 1, 
                  training_epoch =50, entropy_factor_max = 0.1, entropy_factor_min = 0.0001,
                  memory_size = 50):
        
        self.writer = writer
        self.hyper_dir = hyper_dir 
        self.evaluation_epoch = evaluation_epoch
        self.exploration_tech = exploration_tech
        self.training_steps = training_steps
        self.sucess_criteria_epochs = sucess_criteria_epochs
        self.sucess_criteria_value = sucess_criteria_value

        self.discount_min = discount_min
        self.discount_max = discount_max
        self.lr_actor_min = lr_actor_min
        self.lr_actor_max = lr_actor_max
        

        self.dense_min = dense_min
        self.dense_max = dense_max
        self.environment_name  = environment_name

        self.reward_scaler = reward_scaler
        self.num_layers = num_layers_act

        self.gae_min = gae_min
        self.gae_max = gae_max
        self.training_epoch = training_epoch
        self.memory_size = memory_size


        self.entropy_factor_min = entropy_factor_min
        self.entropy_factor_max = entropy_factor_max

    def build(self, hp):
        
        discount =  hp.Float('discount',self.discount_min, self.discount_max, step =0.01)
        gae_lambda =  hp.Float('gae_lambda',self.gae_min, self.gae_max, step =0.01)
        learning_rate = hp.Float('learning_rate', self.lr_actor_min, self.lr_actor_max)
        policy_clip = hp.Float('policy_clip', 0.1, 0.3, step =0.1)
        entropy_coeff = hp.Float('entropy_coeff', self.entropy_factor_min, self.entropy_factor_max)

        if self.num_layers>1 :
            num_layer = hp.Int('n_dense_layers', 1, self.num_layers)
        else : 
            num_layer = self.num_layers
        dense_units =  [hp.Int('dense_units_'+str(i), self.dense_min, self.dense_max, step =32) for i in range(num_layer)]


        actorcritic_agent = PPO(
            discount = discount, 
            dense_units = dense_units,
            num_layer  = num_layer, 
            writer = self.writer,
            trial_n = get_valid_trials_number(self.hyper_dir ),
            evaluation_epoch = self.evaluation_epoch,
            environment_name = self.environment_name,
            reward_scaler = self.reward_scaler,
            learning_rate = learning_rate,
            gae_lambda=gae_lambda,
            policy_clip = policy_clip,
            training_epoch = self.training_epoch,
            entropy_coeff = entropy_coeff,
            memory_size = self.memory_size
            )
        return actorcritic_agent

    def fit(self, hp, model, x, y,  callbacks=None, **kwargs):
        
        training_steps = self.training_steps
        
        # Record the best validation loss value
        best_epoch_loss = float(-100000)
        
        # Assign the model to the callbacks.rewards_val_history
        for callback in callbacks:
            callback.model = model

        
        obs = model.train_env.reset()
        for epoch in range(training_steps):
            next_obs = model.run_agent(epoch, obs)
            model.evaluate_agent(epoch)

            obs = next_obs
            if epoch %  model.evaluation_epoch == 0 and len(model.rewards_val_history)> 0 and len(model.rewards_train_history)> 0 and epoch >0:
                print(f"Epoch: {epoch} : Reward eval/Train: {np.mean(model.rewards_val_history)}/{np.mean(model.rewards_train_history)} ")

            if len(model.rewards_val_history)> self.sucess_criteria_epochs :
                if np.mean(model.rewards_val_history[-self.sucess_criteria_epochs:]) >= self.sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break
        
        final_reward = np.mean(model.rewards_train_history)
        best_epoch_loss = max(best_epoch_loss, final_reward)

        for callback in callbacks:
            # The "my_metric" is the objective passed to the tuner.
            callback.on_epoch_end(epoch, logs={"total_train_reward" : final_reward})#total_eval_reward

        # Return the evaluation metric value.
        return best_epoch_loss




def run_training(training_steps,   discount,  dense_units,num_layers,
                  writer,  save_factor=50000, sucess_criteria_epochs =100, sucess_criteria_value = -100, 
                 environment_name="MountainCar-v0", reward_scaler = 1, evaluation_epoch = 2000,return_agent = False,
                 lr= 0.001,   gae_lambda=0.95, training_epoch= 20, entropy_coeff= 0.01, policy_clip = 0.2,memory_size= 50):
  
    model = PPO(
            discount = discount, 
            dense_units = dense_units,
            num_layer  = num_layers, 
            writer = writer,
            trial_n = get_valid_trials_number(writer ),
            evaluation_epoch = evaluation_epoch,
            environment_name = environment_name,
            reward_scaler = reward_scaler,
            learning_rate = lr,
            gae_lambda=gae_lambda,
            policy_clip = policy_clip,
            training_epoch = training_epoch ,
            entropy_coeff = entropy_coeff,
            memory_size = memory_size
            )
    
    
    with tqdm.trange(training_steps) as t:
        obs = model.train_env.reset()
        for epoch in t:
            new_obs = model.run_agent(epoch, obs)
            model.evaluate_agent(epoch)
            
            obs = new_obs
            if epoch %save_factor == 0: 
                model.save_weights('./checkpoints/PPOagent')

            if len(model.rewards_train_history)> 100 :
                if epoch %  model.evaluation_epoch*5 == 0 :
                    print(f"Epoch: {epoch} : Reward Train: {np.mean(model.rewards_train_history[-100:])} ")

                if np.mean(model.rewards_val_history[-sucess_criteria_epochs:]) >= sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break

    if return_agent:
        return model

def final_evaluation(eval_model, eval_env, n_tries=1, exploration ="soft", video_name = "./PPO_soft_video.mp4", sucess_criteria_epochs= 100, reward_scaler = 1, env= ""):
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
            actions,_,_ = eval_model.actor_critic.get_action(obs.reshape((1,eval_model.obs_shape[0])))
            if env== "BipedalWalker-v3":
                actions = actions[0,]
                
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
