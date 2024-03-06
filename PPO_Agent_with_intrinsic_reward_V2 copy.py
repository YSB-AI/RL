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
from keras import backend as K


import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder


print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)


class PPO(tf.keras.Model):
    def __init__(self,  discount, dense_units_act_crit,  dense_units_model, num_layer_act_crit, num_layer_model, writer,  lr_actor_critic, lr_model, trial_n = "", 
    evaluation_epoch = 2500, environment_name="",  gae_lambda=0.95, policy_clip = 0.2, training_epoch=20, entropy_coeff = 0.05, memory_size = 50, scaling_factor_reward = 0.1,
    normalize = False):
        super(PPO, self).__init__()
        
        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.evaluation_epoch = evaluation_epoch
        self.environment_name= environment_name
        self.memory_size = memory_size
        
        self.memory = deque(maxlen= self.memory_size)
        self.episode_reward = 0 
        self.training_epoch = training_epoch
        self.training_counter = 0 
        self.exploration_update = 5000
        
        # Training parameters
        
        self.scaling_factor  = scaling_factor_reward
        self.policy_clip = policy_clip
        self.entropy_coeff = entropy_coeff
        
        self.gae_lambda=gae_lambda
        self.discount = discount
        self.normalize = normalize
        
        # Optimizers and loss functions        
        self.actor_critic_optimizer=Adam(learning_rate=lr_actor_critic)
        self.env_model_optimizer = Adam(learning_rate=lr_model)
        
        self.env_model_loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Agents components
        self.actor_critic = Actor_Critic(env = self.env,  d = dense_units_act_crit, num_layer_act_crit = num_layer_act_crit)
        self.env_model = EnvironmentModel(env = self.env, obs_shape=self.obs_shape[0],  d = dense_units_model,  num_layer_model = num_layer_model)

        self.train_env =  gym.make(environment_name)
        
        # Logs parameters
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('discount', discount, step=0)
            tf.summary.scalar('gae_lambda', gae_lambda, step=0)
            for i in range(num_layer_act_crit):
                tf.summary.scalar('dense_units_actor_critic_'+str(i), dense_units_act_crit[i], step=0)

            tf.summary.text('environment_name', self.environment_name , step=0)
            tf.summary.scalar('lr_actor_critic', lr_actor_critic , step=0)


        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewars = 0


    def get_sample(self,next_state_value):
        s, s_, a, policy, v, rewards, dones = [], [], [], [], [], [], []
        
        #[obs, next_obs, action, log_probs, state_value, reward, done,])
        for each in self.memory: 

            s.append(each[0].reshape((1,self.obs_shape[0])))
            s_.append(each[1].reshape((1,self.obs_shape[0])))
            a.append(each[2])
            policy.append(each[3])
            v.append(each[4])
            rewards.append(each[5])
            dones.append( 1 - int(each[6]))
        
        if self.normalize : 
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)+ 1e-7
            normalized_rewards = [(r - mean_reward) / std_reward for r in rewards]
            rewards = normalized_rewards

        v.append(next_state_value)
        
        s = np.array(s).reshape((len(self.memory), self.obs_shape[0]))
        s_ = np.array(s_).reshape((len(self.memory), self.obs_shape[0]))
        
        if self.env.action_space.shape[0] > 1:
            a = np.array(a)
        else:
            a = np.array(a).reshape((len(a)))

        #log_prob = np.array(log_prob)
        a =  tf.cast(tf.convert_to_tensor(a), dtype = tf.float32)
        v =  tf.cast(tf.convert_to_tensor(v), dtype = tf.float32)#np.array(v)
        policy=  tf.cast(tf.convert_to_tensor(policy), dtype = tf.float32)
        rewards= tf.cast(tf.convert_to_tensor(rewards), dtype = tf.float32)#np.array(rewards)
        dones = tf.cast(tf.convert_to_tensor(dones), dtype = tf.float32) #np.array(dones)

        return s, s_, a, policy, v,rewards,dones
    
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def process_training(self,obs, next_obs, actions, returns, advantages, old_policy):
        
        old_policy = tf.reshape(old_policy, (len(actions),self.env.action_space.shape[0]))
        with tf.GradientTape() as tape:
            
            # Update actor network
            _, state_value, policy, log_probs, entropy  = self.actor_critic(obs, training=True)
            log_probs = tf.reshape(log_probs, (len(actions),self.env.action_space.shape[0]))

            prob_ratio = tf.math.exp(log_probs - old_policy) 
            
            # if prob_ratio.shape[1]> 1 :
            #     prob_ratio = tf.reduce_mean(prob_ratio, axis = -1)
            weighted_clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
            
            #Update critic network
            # print("obs ->",obs.shape)
            # print("next_obs ->",next_obs.shape)
            # print("actions ->",actions.shape)
            # print("old_policy ->",old_policy.shape)
            # print("policy ->",policy)
            
            # print("advantages ->",advantages.shape)
            # print("prob_ratio ->",prob_ratio.shape)
            # print("weighted_clipped_probs ->",weighted_clipped_probs.shape)
            
            # print("state_value ->",state_value.shape)
            # print("returns ->",returns.shape)
            
            #Entropy regularization
            #entropy = -tf.reduce_mean(policy * log_probs)
            # print("entropy ->",entropy.shape)
            
            critic_loss =tf.reduce_mean((state_value - returns)**2, axis=0) *0.5
            clipped_surr_actor_loss = -tf.reduce_mean(tf.minimum(prob_ratio * advantages, weighted_clipped_probs * advantages)) 

            total_loss = clipped_surr_actor_loss + critic_loss - self.entropy_coeff * entropy 
            
        actor_critic_grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic_optimizer.apply_gradients(zip(actor_critic_grads, self.actor_critic.trainable_variables))
        
        # print("critic_loss ->",critic_loss.shape)
        # print("clipped_surr_actor_loss ->",clipped_surr_actor_loss.shape)
        
        return total_loss, clipped_surr_actor_loss, critic_loss, state_value, entropy, prob_ratio, actor_critic_grads
    
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def training_env_model(self, obs, next_obs, action_arr):
        
        with tf.GradientTape() as tape:
            predicted_next_obs = self.env_model(obs,tf.reshape(action_arr, (len(action_arr),self.env.action_space.shape[0])),)
            model_loss = self.env_model_loss_fn(next_obs, predicted_next_obs)

        model_grads = tape.gradient(model_loss, self.env_model.trainable_variables)
        self.env_model_optimizer.apply_gradients(zip(model_grads, self.env_model.trainable_variables))
        
        return model_loss, model_grads
    
    
    # def get_advantage_and_return(self, rewards, state_values,final_value, dones):
    #     returns = np.zeros_like(rewards)
    #     advs = np.zeros_like(rewards)
    #     last_gae_lam = 0

    #     num_steps = len(rewards)
        
    #     for t in reversed(range(num_steps)):
    #         if t == num_steps - 1:
    #             next_non_terminal = 0
    #             next_values = final_value
    #         else:
    #             next_non_terminal = 1.0 - dones[t + 1]
    #             next_values = state_values[t + 1]
            
    #         delta = rewards[t] + self.discount * next_values * next_non_terminal - state_values[t]
    #         advs[t] = last_gae_lam = delta + self.discount * self.gae_lambda * next_non_terminal * last_gae_lam
        
        
    #     returns = advs + state_values                         # ADV = RETURNS - VALUES
    #     advs = (advs - advs.mean()) / (advs.std())      # Normalize ADVs
        
    #     return advs, returns

    
    def train(self, next_state_value, epoch):
        
        obs, next_obs, actions, old_policy, state_values, rewards, dones =  self.get_sample(next_state_value)
        
        final_value = state_values[-1] 
        state_values = state_values[:-1] 
        #advantages, returns  = self.get_advantage_and_return(rewards, state_value, final_value, dones)
        
        
        rewards = tf.cast(tf.reshape(rewards, (len(rewards),1)), dtype = tf.float32)
        final_value = tf.cast(tf.reshape(final_value, (1,)), dtype = tf.float32)
        state_values = tf.cast(tf.reshape(state_values, (len(state_values),1)), dtype = tf.float32)

        discounts = self.discount *  dones
        discounts = tf.cast(np.array(discounts).reshape((len(discounts),1)), dtype = tf.float32)
        
        

        # print("rewards ->",rewards)
        # print("final_value ->",final_value)
        # print("state_values ->",state_values)
        # print("discounts ->",discounts)

        advantages = tf_agents.utils.value_ops.generalized_advantage_estimation(
            values = state_values, final_value = final_value, discounts =  discounts, 
            rewards = rewards, td_lambda=self.gae_lambda, time_major=True
        )
        
        returns=  tf_agents.utils.value_ops.discounted_return(rewards,discounts,time_major=True, final_value=final_value, provide_all_returns = True) 
        returns = tf.squeeze(returns, axis = -1)


        # advantage_mean =  tf.reduce_mean(advantages, axis=0)
        # advantage_std =  tf.math.reduce_std(advantages, axis=0) + 1e-10
        # advantages = (advantages - advantage_mean)/advantage_std
        # advantages= tf.cast(tf.convert_to_tensor(advantages), dtype = tf.float32)
        
        # rewards = tf.cast(tf.convert_to_tensor(rewards), dtype = tf.float32)
        # final_value = tf.cast(tf.convert_to_tensor(final_value), dtype = tf.float32)
        # state_values = tf.cast(tf.convert_to_tensor(state_values), dtype = tf.float32)

        #advantages= tf.cast(tf.convert_to_tensor(advantages), dtype = tf.float32)
        #returns= tf.cast(tf.convert_to_tensor(returns), dtype = tf.float32)
        
        obs = tf.cast(tf.convert_to_tensor(obs), dtype = tf.float32)
        next_obs  = tf.cast(tf.convert_to_tensor(next_obs), dtype = tf.float32)
        old_policy = tf.cast(tf.convert_to_tensor(old_policy), dtype = tf.float32)
        
        for _ in range(self.training_epoch):
            total_loss, actor_loss, critic_loss, state_value, entropy, prob_ratio, actor_critic_grads = self.process_training(obs, next_obs, actions, returns, advantages, old_policy)
            model_loss, model_grads = self.training_env_model(obs, next_obs, actions)
        
             
        with self.tb_summary_writer.as_default():
            for i, grad in enumerate(actor_critic_grads):
                tf.summary.histogram(f'ActorCritic/Gradients/Layer_{i}', grad, step=self.training_counter )
                
            for i, grad in enumerate(model_grads):
                tf.summary.histogram(f'Env/Gradients/Layer_{i}', grad, step=self.training_counter )
            
   
        self.training_counter  = self.training_counter  +1
        
        return total_loss, actor_loss, critic_loss, advantages, state_value, returns, entropy, prob_ratio, model_loss
  
    
    def calculate_intrinsic_reward(self, obs, action, next_obs):
        predicted_next_obs = self.env_model(obs, action)
        predicted_next_obs = predicted_next_obs.numpy()
        prediction_error = np.mean(np.square(predicted_next_obs - next_obs))

        return prediction_error, predicted_next_obs[0]


    def run_agent(self, epoch, obs):
        action, state_value, pi, log_probs,_  = self.actor_critic(obs.reshape((1,self.obs_shape[0])), inference=True)

        # if self.environment_name in ["BipedalWalker-v3", "Ant-v2"]:
        #     action = action[0,]
        next_obs, reward, done, info = self.train_env.step(action)
        intrinsic_reward, predicted_next_obs = self.calculate_intrinsic_reward( obs.reshape((1,self.obs_shape[0])), # REVIEW 
                                                           tf.reshape(action, shape =(1,self.env.action_space.shape[0]) ),
                                                           next_obs.reshape((1,self.obs_shape[0])))
        
        total_reward = reward + self.scaling_factor * intrinsic_reward
        #done = truncated or terminated  #terminated, truncated,
        self.episode_reward += reward

        
        self.memory.append([obs, next_obs, action, log_probs,  state_value.numpy(), total_reward, done])
        self.append_interaction_metrics(action, reward,total_reward, done, epoch)
        
        if done or len(self.memory)== self.memory_size:        
            
            _,next_state_value, _, _,_ = self.actor_critic(next_obs.reshape((1,self.obs_shape[0])), inference=True)
            
            total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, model_loss = self.train(next_state_value, epoch)
            
            # print("advantages ->", advantages)
            # print("state_values ->", state_values)
            # print("returns ->", returns)
            # print("entropy ->", entropy)
            # print("prob_ratio ->", prob_ratio)
            # print("prob_ratio ->", model_loss)
            # print("total_loss ->", total_loss)
            # print("actor_loss ->", actor_loss)
            # print("critic_loss ->", critic_loss)
            
            
            self.memory.clear()
                
            if done :
                self.append_training_metrics(self.episode_reward, total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, model_loss, epoch)
                self.rewards_train_history.append(self.episode_reward)
                self.episode_reward = 0 
                next_obs = self.train_env.reset()
                predicted_next_obs = next_obs

        return predicted_next_obs#next_obs#predicted_next_obs
        

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()
            total_reward = 0
            while(True):
                
                
                actions,_, _, _ , _  = self.actor_critic(obs.reshape((1,self.obs_shape[0])), inference=True)
                # if self.environment_name in ["BipedalWalker-v3", "Ant-v2"]:
                #     actions = actions[0,]
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
            rewards_history = np.mean(self.evaluate(gym.make(self.environment_name) , n_tries=1))
            self.rewards_val_history.append(rewards_history)

            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', self.rewards_val_history[-1], step=epoch)

                if len(self.rewards_val_history)> sucess_criteria_epochs: 
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-sucess_criteria_epochs:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-sucess_criteria_epochs:]), step=epoch)
    


    def append_interaction_metrics(self, action, reward,  total_reward, done, epoch):

        with self.tb_summary_writer.as_default():
            tf.summary.histogram('Action', action, step=epoch)
            tf.summary.scalar('RT_Reward', reward, step=epoch)
            tf.summary.scalar('Intrinsic_Reward', total_reward, step=epoch)
            
            if done == 1:
                tf.summary.scalar('Done', 1, step=epoch)
                tf.summary.scalar('Done', 0, step=epoch+1)
            

    def append_training_metrics(self, episode_reward, total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, model_loss, epoch):

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('Episode_Reward', episode_reward, step=epoch)
            tf.summary.scalar('Entropy', tf.reduce_mean((entropy)), step=epoch)
            tf.summary.scalar('total_loss', tf.reduce_mean((total_loss)), step=epoch)
            tf.summary.scalar('actor_loss', tf.reduce_mean((actor_loss)), step=epoch)
            tf.summary.scalar('critic_loss', tf.reduce_mean((critic_loss)), step=epoch)
            if model_loss != None : tf.summary.scalar('model_loss', model_loss, step=epoch)
            tf.summary.scalar('Advantages', tf.reduce_mean((advantages)), step=epoch)
            tf.summary.scalar('State_Value', tf.reduce_mean(tf.convert_to_tensor(state_values)), step=epoch)
            tf.summary.scalar('Expected_Return', tf.reduce_mean(tf.convert_to_tensor(returns)), step=epoch)
            tf.summary.scalar('Prob_ratio', tf.reduce_mean(tf.convert_to_tensor(prob_ratio)), step=epoch)

class Actor_Critic(tf.keras.Model):
    def __init__(self,  env,  d, num_layer_act_crit = 1):
        super(Actor_Critic, self).__init__()
        self.env = env
        self.num_layers = num_layer_act_crit
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_act_crit)]
        self.d_state_value = Dense(1,  name ="v")

        self.noise= 1e-7
        self.mean_dense = Dense(units=self.env.action_space.shape[0], name='actor_mu' )
        self.log_std_dense = Dense(units=self.env.action_space.shape[0],  name='actor_sigma')
        self.noise= 1e-7

    def call(self, observations, inference = False):

        # Get mean and standard deviation from the policy network
        x = observations
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
        
        state_value = self.d_state_value(x)
        state_value = tf.squeeze(self.d_state_value(x))
        
        mu = self.mean_dense(x) 
        log_std = self.log_std_dense(x)
        log_std = tf.clip_by_value(log_std, -20, 10)
        std = tf.exp(log_std)
        
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        
        #Gaussian**********
        pre_sum = -0.5 * (((pi-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
        logp_pi = tf.reduce_sum(pre_sum, axis=1)
        #Squashing Function**********
        logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
        # Squash those unbounded actions!
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        #***********************************
        
        action_scale = self.env.action_space.high[0]
        mu *= action_scale
        pi *= action_scale
        
        #normal_dist = tfp.distributions.Normal(mu, std)
        #entropy = -normal_dist.entropy()
        entropy =  0.5 * tf.math.log(2 * np.pi * np.e * std**2)
        return mu, state_value, pi, logp_pi, entropy
    
# class Actor_Critic(tf.keras.Model):
#     def __init__(self,  env,  d, num_layer_act_crit = 1):
#         super(Actor_Critic, self).__init__()
#         self.env = env
#         self.num_layers = num_layer_act_crit
#         self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_act_crit)]
#         self.d_state_value = Dense(1,  name ="v")

#         self.noise= 1e-7
#         self.mean_dense = Dense(units=self.env.action_space.shape[0], name='actor_mu' )
#         self.log_std_dense = Dense(units=self.env.action_space.shape[0],  name='actor_sigma')

#     def call(self, observations, inference = False):

#         # Get mean and standard deviation from the policy network
#         x = observations
#         for i in range(self.num_layers):
#             x = self.dense_layers[i](x)
        
#         state_value = self.d_state_value(x)
#         state_value = tf.squeeze(self.d_state_value(x))
        
#         mu = self.mean_dense(x) 
#         log_std = self.log_std_dense(x)
#         log_std = tf.clip_by_value(log_std, -10, 2)
        
#         std = tf.exp(log_std)       # Take exp. of Std deviation
        
#         action = mu + tf.random.normal(tf.shape(mu)) * std                      # Sample action from Gaussian Dist
        
#         action = tf.clip_by_value(action, self.env.action_space.low[0], self.env.action_space.high[0])                                # Clip Continuous actions in range of [-1, 1] 
#         log_probs = self.gaussian_likelihood(action, mu, log_std)                                          # Calculate logp at timestep t for actions

#         pi = tf.math.exp(log_probs)
#         entropy = self.entropy(log_std)
#         return action,  state_value, pi, log_probs,entropy
    

#     def gaussian_likelihood(self, x, mu, log_std):
#         """
#             calculate the liklihood logp of a gaussian distribution 
#             for parameters x given the variables mu and log_std
#         """
#         pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + self.noise))**2 + 2 * log_std + np.log(2 * np.pi))
#         return tf.reduce_sum(pre_sum, axis= -1)

#     def entropy(self, log_std):
#         """
#             Entropy term for more randomness 
#             which means more exploration in ppo 
#         """       38359
#         entropy = tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
#         return entropy
    

# class Actor_Critic(tf.keras.Model):
#     def __init__(self,  env,  d, num_layer_act_crit = 1):
#         super(Actor_Critic, self).__init__()
#         self.env = env
#         self.num_layers = num_layer_act_crit
#         self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_act_crit)]
#         self.d_state_value = Dense(1,  name ="v")

#         self.noise= 1e-7
#         #self.mean_dense = Dense(units=self.env.action_space.shape[0], name='actor_mu' )
#         #self.log_std_dense = Dense(units=self.env.action_space.shape[0],  name='actor_sigma')
#         self.mean_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.tanh, name='actor_mu' )
#         self.log_std_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.softplus, name='actor_sigma' ) 
#         self.noise= 1e-7

#     def call(self, observations, inference = False):

#         # Get mean and standard deviation from the policy network
#         x = observations
#         for i in range(self.num_layers):
#             x = self.dense_layers[i](x)
        
#         state_value = self.d_state_value(x)
        
#         mu = self.mean_dense(x) 
#         std = self.log_std_dense(x)
        
#         mu, std = tf.squeeze(mu*self.env.action_space.high[0]), tf.squeeze(std + self.noise)
#         state_value = tf.squeeze(self.d_state_value(x))

#         normal_dist = tfp.distributions.Normal(mu, std)
#         action = tf.clip_by_value(normal_dist.sample(1) , self.env.action_space.low[0], self.env.action_space.high[0])
#         log_probs = tf.cast(normal_dist.log_prob(action), dtype = tf.float32)
#         entropy = normal_dist.entropy()
#         pi = tf.math.exp(log_probs)
        
#         return action, state_value, pi,  log_probs, entropy
  


class EnvironmentModel(tf.keras.Model):
    def __init__(self,  env, obs_shape, d,  num_layer_model = 1):
        super(EnvironmentModel, self).__init__()
        self.env = env
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_model)]
        self.num_layers = num_layer_model
        self.next_state = Dense(obs_shape,  name ="next_state")
        
    def call(self, observations, action):
        
        x = tf.concat([observations, action], axis=-1)
         
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
        next_state = self.next_state(x)
        
        next_state = tf.clip_by_value(next_state , self.env.observation_space.low[0], self.env.observation_space.high[0])
        return next_state

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
                  discount = None,
                  policy_clip = None,
                  gae_min = 0.90, gae_max = 0.99,
                  lr_actor_min = 0.00001, lr_actor_max = 0.005,
                  lr_critic_min = 0.00001, lr_critic_max = 0.005,
                  lr_model_min = 0.000001, lr_model_max = 0.001,
                  dense_min = 32, dense_max = 200,
                  environment_name="MountainCar-v0" , 
                  num_layers_act = 1, num_layers_crit = 1, num_layers_model = 1, 
                  training_epoch =50, entropy_factor_max = 0.1, entropy_factor_min = 0.0001,
                  memory_size = 50, training_epoch_max = None, memory_size_max = None, normalize = False, scaling_factor_reward = None):
        
        self.writer = writer
        self.hyper_dir = hyper_dir 
        self.evaluation_epoch = evaluation_epoch
        self.exploration_tech = exploration_tech
        self.training_steps = training_steps
        self.sucess_criteria_epochs = sucess_criteria_epochs
        self.sucess_criteria_value = sucess_criteria_value
        self.training_epoch_max = training_epoch_max 
        self.memory_size_max = memory_size_max 

        self.discount_min = discount_min
        self.discount_max = discount_max
        
        self.lr_actor_min = lr_actor_min
        self.lr_actor_max = lr_actor_max
        
       
        self.lr_model_min = lr_model_min
        self.lr_model_max = lr_model_max

        self.dense_min = dense_min
        self.dense_max = dense_max
        self.environment_name  = environment_name


        self.num_layers_act = num_layers_act
        self.num_layers_crit = num_layers_crit
        

        self.gae_min = gae_min
        self.gae_max = gae_max
        self.training_epoch = training_epoch
        self.memory_size = memory_size

        self.entropy_factor_min = entropy_factor_min
        self.entropy_factor_max = entropy_factor_max
        
        self.num_layers_model  = num_layers_model
        
        self.discount = discount
        
        self.policy_clip = policy_clip
        self.normalize_rewards = normalize
        
        self.scaling_factor_reward = scaling_factor_reward

    def build(self, hp):
        K.clear_session()

        if self.training_epoch_max != None:
            self.training_epoch = hp.Int('training_epoch', 10, self.training_epoch_max, step = 10)
            
        if self.memory_size_max != None:
            self.memory_size = hp.Int('memory_size_max', 10, self.memory_size_max, step = 8)
        
        
        discount = self.discount
        if discount is None: discount =  hp.Float('discount',self.discount_min, self.discount_max, step =0.01)
        
        gae_lambda =  hp.Float('gae_lambda',self.gae_min, self.gae_max, step =0.01)
        lr_actor_critic = hp.Float('lr_actor_critic', self.lr_actor_min, self.lr_actor_max)
        lr_model = hp.Float('lr_model', self.lr_model_min, self.lr_model_max)
        
        policy_clip = self.policy_clip
        if policy_clip is None: policy_clip = hp.Float('policy_clip', 0.1, 0.3, step =0.1)
        
        scaling_factor_reward = self.scaling_factor_reward 
        if self.scaling_factor_reward == None:
            scaling_factor_reward = hp.Float('scaling_factor_reward', 0.01, 0.3)
        entropy_coeff = hp.Float('entropy_coeff', self.entropy_factor_min, self.entropy_factor_max)

        if self.num_layers_act>1 :
            num_layer_act_crit = hp.Int('n_dense_layers_actor', 1, self.num_layers_act)
        else : 
            num_layer_act_crit = self.num_layers_act
        dense_units_act_crit =  [hp.Int('dense_units_act_crit_'+str(i), self.dense_min, self.dense_max) for i in range(num_layer_act_crit)]
       
        if self.num_layers_model>1 :
            num_layer_m = hp.Int('n_dense_layers_model', 1, self.num_layers_model)
        else : 
            num_layer_m = self.num_layers_model
        dense_units_model =  [hp.Int('n_dense_layers_model'+str(i), self.dense_min, self.dense_max) for i in range(num_layer_m)]

        actorcritic_agent = PPO(
            discount = discount, 
            dense_units_act_crit = dense_units_act_crit,
            dense_units_model = dense_units_model,
            num_layer_act_crit  = num_layer_act_crit, 
            num_layer_model = num_layer_m,
            writer = self.writer,
            trial_n = get_valid_trials_number(self.hyper_dir ),
            evaluation_epoch = self.evaluation_epoch,
            environment_name = self.environment_name,
            lr_actor_critic = lr_actor_critic,  lr_model = lr_model, 
            gae_lambda=gae_lambda,
            policy_clip = policy_clip,
            training_epoch = self.training_epoch,
            entropy_coeff = entropy_coeff,
            memory_size = self.memory_size,
            scaling_factor_reward = scaling_factor_reward,
            normalize = self.normalize_rewards
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
            if epoch %  2000 == 0 and len(model.rewards_val_history)> 0 and len(model.rewards_train_history)> 0 and epoch >0:
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




def run_training(training_steps,   discount,  dense_units_act,  dense_units_crit,dense_units_model,num_layer_a,num_layer_c,num_layer_m,
                  writer,  save_factor=50000, sucess_criteria_epochs =100, sucess_criteria_value = -100, 
                 environment_name="MountainCar-v0",  evaluation_epoch = 2000,return_agent = False,
                 lr_actor= 0.001, lr_critic = 0.001,  lr_model= 0.00001,  gae_lambda=0.95, training_epoch= 20, entropy_coeff= 0.01, 
                 policy_clip = 0.2,memory_size= 50, id = 1, normalize = False, scaling_factor_reward = 0.1):
  
    model = PPO(
            discount = discount, 
            dense_units_act_crit = dense_units_act,
            dense_units_model = dense_units_model, 
            num_layer_act_crit  = num_layer_a, 
            num_layer_model = num_layer_m,
            writer = writer,
            trial_n = get_valid_trials_number(writer ),
            evaluation_epoch = evaluation_epoch,
            environment_name = environment_name,
            lr_actor_critic = lr_actor,lr_model=lr_model,
            gae_lambda=gae_lambda,
            policy_clip = policy_clip,
            training_epoch = training_epoch ,
            entropy_coeff = entropy_coeff,
            memory_size = memory_size,
            normalize = normalize,
            scaling_factor_reward=scaling_factor_reward
            )
    
    
    with tqdm.trange(training_steps) as t:
        obs = model.train_env.reset()
        for epoch in t:
            new_obs = model.run_agent(epoch, obs)
            model.evaluate_agent(epoch)
            
            obs = new_obs
            if epoch %save_factor == 0: 
                model.save_weights('./checkpoints/PPOagent'+str(id))

            if len(model.rewards_train_history)> 100 :
                if epoch %  model.evaluation_epoch*5 == 0 :
                    print(f"Epoch: {epoch} : Reward Train: {np.mean(model.rewards_train_history[-100:])} ")

            if np.mean(model.rewards_val_history[-sucess_criteria_epochs:]) >= sucess_criteria_value:
                print("Your agent reached the objetive")
                break

    if return_agent:
        return model

def final_evaluation(eval_model, eval_env, n_tries=1, exploration ="soft", video_name = "./PPO_soft_video.mp4", sucess_criteria_epochs= 100):
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
            actions, mu, sigma , log_probs, entropy  = eval_model.actor_critic(obs.reshape((1,eval_model.obs_shape[0])), inference=True)
                     
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
