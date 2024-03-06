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

#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


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
import keras_tuner as kt
import tensorflow_probability as tfp

tf.config.run_functions_eagerly(False) 

import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder

print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)


class SAC(tf.keras.Model):
    def __init__(self,  discount, dense_units_act, dense_units_crit, tau, num_layer_act, num_layer_crit, writer,  lr_actor, lr_critic, lr_alpha, trial_n = "", end_of_episode = 200, 
    evaluation_epoch = 2500, environment_name="", reward_scaler  = 1, alpha = 0.2, train_epochs = 20):
        super(SAC, self).__init__()
        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.evaluation_epoch = evaluation_epoch
        self.environment_name= environment_name
        self.memory = deque(maxlen= 100000)
        self.update_after = 1000
        self.batch_size = 128
        self.update_every = 50
        self.train_epochs= train_epochs

        self.target_entropy = -tf.constant(self.env.action_space.shape[0], dtype=tf.float32)
        self.alpha = tf.Variable(0.0, dtype=tf.float32)

        # Training parameters
        self.end_of_episode = end_of_episode

        self.discount = discount
        self.actor_optimizer=Adam(learning_rate=lr_actor)
        self.critic1_optimizer=Adam(learning_rate=lr_critic)
        self.critic2_optimizer=Adam(learning_rate=lr_critic)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr_alpha, epsilon=1e-04)
        self.reward_scaler = reward_scaler
        self.tau = tau

        self.policy = Actor(env = self.env,  d = dense_units_act, num_layer_act = num_layer_act)
        self.qdouble_head = Critic(env = self.env, d = dense_units_crit,  num_layer_crit = num_layer_crit)
        self.target_qdouble_head = Critic(env = self.env, d = dense_units_crit,  num_layer_crit = num_layer_crit)
        
        self.train_env =  gym.make(environment_name)
        
        # Logs parameters
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)

    

        with self.tb_summary_writer.as_default():
            tf.summary.scalar('discount', discount, step=0)
            for i in range(num_layer_act):
                tf.summary.scalar('dense_units_act_'+str(i), dense_units_act[i], step=0)

            for i in range(num_layer_crit):
                tf.summary.scalar('dense_units_crit_'+str(i), dense_units_crit[i], step=0)

            tf.summary.scalar('end_of_episode', self.end_of_episode , step=0)
            tf.summary.text('environment_name', self.environment_name , step=0)
            tf.summary.scalar('lr_actor', lr_actor , step=0)
            tf.summary.scalar('lr_critic_1', lr_critic , step=0)
            tf.summary.scalar('reward_scaler', reward_scaler , step=0)


        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewards = 0

    def sample_action(self, current_state, save_action = True, epoch = 0):
            current_state_ = np.array(current_state, ndmin=2)
            mu,log_pi_a = self.policy(current_state_)
            action, _,_  = self.get_policy(mu,log_pi_a)

            # if save_action:
            #     with self.tb_summary_writer.as_default():
            #         tf.summary.histogram(f"action", action, epoch)

            return action[0]
    
    def get_sample(self):
        batch = random.sample(self.memory, self.batch_size)

        s = np.array([each[0] for each in batch]).reshape((self.batch_size, self.obs_shape[0]))
        a = np.array([each[1] for each in batch])
        s_ = np.array([each[2] for each in batch]).reshape((self.batch_size, self.obs_shape[0]))
        r = np.array([each[3] for each in batch]).reshape((self.batch_size, 1))
        #r = r - np.mean(r, axis=0) / (np.std(r, axis=0) + 1e-6) 
        dones = np.array([(1-each[4]) for each in batch]).reshape((self.batch_size, 1))
        
    
        return s,a,r,s_,dones
 

    def update_q_network(self, current_states, actions, rewards, next_states, ends, epoch):

        # Sample actions from the policy for next states
        mu, log_pi_a = self.policy(next_states)
        mu, pi_a, log_pi_a  = self.get_policy(mu,log_pi_a)
            
        # Get Q value estimates from target Q network
        q1_target, q2_target = self.target_qdouble_head(next_states, pi_a)
        min_q_target = tf.minimum(q1_target, q2_target)
        # Add the entropy term to get soft Q target
        soft_q_target = min_q_target - self.alpha * log_pi_a
        y = tf.stop_gradient(rewards + self.discount * ends * soft_q_target)
        
        # Get Q value estimates, action used here is from the replay buffer
        q1,q2 = self.qdouble_head(current_states, actions)
        critic1_loss = tf.reduce_mean((q1 - y)**2)
        critic2_loss = tf.reduce_mean((q2 - y)**2)

        return critic1_loss, critic2_loss

    def update_policy_network(self, current_states, epoch):
        
        # Sample actions from the policy for current states
        mu, log_pi_a = self.policy(current_states)
        mu, pi_a, log_pi_a  = self.get_policy(mu,log_pi_a)

        # Get Q value estimates from target Q network
        q1,q2 = self.qdouble_head(current_states, pi_a)

        # Apply the clipped double Q trick
        # Get the minimum Q value of the 2 target networks
        min_q =tf.minimum(q1, q2)
        soft_q = self.alpha * log_pi_a - min_q
        actor_loss = tf.reduce_mean(soft_q)

        return actor_loss, log_pi_a

    def get_policy(self, mu, log_std):
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
        
        return mu, pi, logp_pi


    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def train(self, epoch):

        obs, action, reward, next_obs, done = self.get_sample()
        done = tf.cast(done, dtype= tf.float32)
        reward = tf.cast(reward, dtype= tf.float32)

        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape(persistent = True) as tape:
            # Update policy network weights
            actor_loss, log_pi_a = self.update_policy_network(obs, epoch)
            
            # Update Q network weights
            critic1_loss, critic2_loss = self.update_q_network(obs, action, reward, next_obs, done, epoch)
            critic_loss = critic1_loss +  critic2_loss
        
            alpha_loss = tf.reduce_mean( - self.alpha*(log_pi_a + self.target_entropy))
        
        actor_grads = tape.gradient(actor_loss, self.policy.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.qdouble_head.trainable_variables)
        alpha_grads = tape.gradient(alpha_loss, [self.alpha])
        
        # print("\n------------------------------------")
        # for a_grads in actor_grads:
        #     print(a_grads.shape)
        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.policy.trainable_variables))
        self.critic1_optimizer.apply_gradients(zip(critic_grads,self.qdouble_head.trainable_variables))
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.alpha]))
        
        self.update_weights()

        return actor_loss, critic1_loss, critic2_loss,  alpha_loss
  

    def run_agent(self, epoch, obs):


        if epoch < 10000:
            actions = self.train_env.action_space.sample()

        else:
            actions = self.sample_action(obs, save_action = True, epoch=epoch)


        new_obs, rewards, done, info  = self.train_env.step(actions) #terminated, truncated , 
        #done = truncated or terminated 
        
        self.memory.append([obs, actions, new_obs, rewards/self.reward_scaler, done])
        self.total_rewards = self.total_rewards + rewards

        self.append_RT_metrics(actions, obs, done, epoch)

        if len(self.memory) >= self.update_after and epoch % self.update_every == 0:
            for _ in range(self.train_epochs):
                actor_loss, critic_1_loss, critic_2_loss, alpha_loss = self.train(epoch)
                
            self.append_training_metrics( actor_loss, critic_1_loss, critic_2_loss, alpha_loss, epoch)


        obs = new_obs
        if done and epoch >0  :
            obs = self.train_env.reset()
            self.rewards_train_history.append(self.total_rewards)
        
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Training_rewards', self.total_rewards, step=epoch)

            self.total_rewards = 0

        return obs

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()
            total_reward = 0
            while(True):
            
                actions= self.sample_action(obs)
                obs, reward, done, info = eval_env.step(actions) #terminated, truncated , 
                reward = reward
                total_reward += reward

                #done = truncated or terminated
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

    def append_RT_metrics(self, _batch_actions, _batch_states, _batch_done, epoch):

        if _batch_done:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('RT_done', 0, step=epoch)
                tf.summary.scalar('RT_done', 1, step=epoch+1)

        if epoch %50 == 0:
           with self.tb_summary_writer.as_default():
                tf.summary.scalar('RT_rewards', self.total_rewards, step=epoch)
                tf.summary.scalar('entropy', self.alpha , step=epoch)

    def append_training_metrics(self, _actor_loss, critic_1_loss, critic_2_loss, alpha_loss,  epoch):

        if epoch %50 == 0:
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('RT_rewards', self.total_rewards, step=epoch)
                tf.summary.scalar('actor_loss', _actor_loss, step=epoch)
                tf.summary.scalar('critic_1_loss',  np.mean(critic_1_loss), step=epoch)
                tf.summary.scalar('critic_2_loss',  np.mean(critic_2_loss), step=epoch)
                tf.summary.scalar('Entropy loss',  np.mean(alpha_loss), step=epoch)
                tf.summary.scalar('Alpha', self.alpha , step=epoch)

    
    def update_weights(self, init=False):

        """ assign target_network.weights variables to their respective agent.weights values. """
        
        if init:
            for w_agent, w_target in zip(self.qdouble_head.trainable_weights, self.target_qdouble_head.trainable_weights):
                w_target.assign(w_agent)

        else:
            for w_agent, w_target in zip(self.qdouble_head.trainable_weights, self.target_qdouble_head.trainable_weights):
                w_target.assign(self.tau*w_agent + (1 - self.tau) * w_target)
   

class Actor(tf.keras.Model):
    def __init__(self,  env,  d, num_layer_act = 1):
        super(Actor, self).__init__()
        self.env = env
        self.num_layers = num_layer_act
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_act)]
        self.mean_dense = Dense(units=self.env.action_space.shape[0], name='actor_mu' )
        self.log_std_dense = Dense(units=self.env.action_space.shape[0],  name='actor_sigma')
        self.noise = 1e-6
        
    def call(self, observations):
        # Get mean and standard deviation from the policy network
        x = observations
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
        
        mu = self.mean_dense(x) 
        log_std = self.log_std_dense(x)
        log_std = tf.clip_by_value(log_std, -20, 2)
                
        return mu, log_std


class Critic(tf.keras.Model):
    def __init__(self,  env,  d,  num_layer_crit = 1):
        super(Critic, self).__init__()
        self.env = env
      
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_crit)]
        self.num_layers = num_layer_crit
        self.d_state_value_0  = Dense(32,  name ="v_0")
        self.d_state_value = Dense(1,  name ="v")
        
        
        self.d_state_value_v2_0 = Dense(32,  name ="v2_0")
        self.d_state_value_v2 = Dense(1,  name ="v2")
        
    def call(self, observations, action):

        x = tf.concat([observations, action], axis=-1)
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)

        q = self.d_state_value_0(x)
        q = self.d_state_value(q)
        
        
        q2 = self.d_state_value_v2_0(x)
        q2 = self.d_state_value_v2(q2)
        
        return q, q2
    
    
def get_valid_trials_number(hyper_dir):
    
    dir = "./"+hyper_dir
    trial_n = 0 
    if os.path.isdir(dir) :
        trial_n = len(next(os.walk(dir))[1])
        print("Trial number : ",trial_n)
        
    return str(trial_n)+"_"

class MyHyperModel(kt.HyperModel):
    def __init__(self, hyper_dir, writer = "logs_hyper/A3C/" , exploration_tech=None,
                  end_of_episode = 1000, 
                  evaluation_epoch = 2000, training_steps = 600000,
                  sucess_criteria_epochs = 100, sucess_criteria_value= -100,
                  discount_min = 0.96, discount_max = 0.99,
                  lr_actor_min = 0.00001, lr_actor_max = 0.005,
                  lr_critic_1_min = 0.00001, lr_critic_1_max = 0.005,
                  dense_min = 32, dense_max = 200,
                  environment_name="MountainCar-v0" , 
                  reward_scaler = 1, 
                  tau_min = 0.0001, tau_max = 0.1, num_layers_act = 1, num_layers_crit = 1, train_epochs=20):
        
        self.writer = writer
        self.hyper_dir = hyper_dir 
        self.evaluation_epoch = evaluation_epoch
        self.exploration_tech = exploration_tech
        self.training_steps = training_steps
        self.sucess_criteria_epochs = sucess_criteria_epochs
        self.sucess_criteria_value = sucess_criteria_value

        self.end_of_episode = end_of_episode
        self.discount_min = discount_min
        self.discount_max = discount_max
        self.lr_actor_min = lr_actor_min
        self.lr_actor_max = lr_actor_max
        
        self.lr_critic_1_min = lr_critic_1_min
        self.lr_critic_1_max = lr_critic_1_max

        self.dense_min = dense_min
        self.dense_max = dense_max
        self.environment_name  = environment_name

        self.tau_min = tau_min
        self.tau_max = tau_max

        self.reward_scaler = reward_scaler
        self.num_layers_act = num_layers_act
        self.num_layers_crit = num_layers_crit
        self.train_epochs = train_epochs

    def build(self, hp):
        
        end_of_episode = self.end_of_episode
        discount  = 0.99#hp.Float('discount',self.discount_min, self.discount_max, step =0.01)
        lr_actor = hp.Float('lr_actor', self.lr_actor_min, self.lr_actor_max)
        lr_critic_1 = hp.Float('lr_critic_1', self.lr_critic_1_min, self.lr_critic_1_max)
        lr_alpha = hp.Float('lr_alpha', self.lr_critic_1_min, self.lr_critic_1_max)
        tau = hp.Float('tau', self.tau_min, self.tau_max)


        if self.num_layers_act>1 :
            num_layer_a = hp.Int('n_dense_layers_actor', 1, self.num_layers_act)
        else : 
            num_layer_a = self.num_layers_act
        dense_units_act =  [hp.Int('dense_units_act_'+str(i), self.dense_min, self.dense_max) for i in range(num_layer_a)]

        if self.num_layers_crit>1 :
            num_layer_c = hp.Int('n_dense_layers_critic', 1, self.num_layers_crit)
        else : 
            num_layer_c = self.num_layers_crit

        dense_units_crit =  [hp.Int('dense_units_crit_'+str(i), self.dense_min, self.dense_max) for i in range(num_layer_c)]

        actorcritic_agent = SAC(
            discount = discount, 
            dense_units_act = dense_units_act,
            dense_units_crit= dense_units_crit, 
            num_layer_act  = num_layer_a, 
            num_layer_crit= num_layer_c,
            writer = self.writer,
            trial_n = get_valid_trials_number(self.hyper_dir ),
            end_of_episode = end_of_episode,
            evaluation_epoch = self.evaluation_epoch,
            environment_name = self.environment_name,
            reward_scaler = self.reward_scaler,
            lr_actor = lr_actor, lr_critic = lr_critic_1, lr_alpha = lr_alpha, 
            tau= tau,
            train_epochs = self.train_epochs
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
        model.update_weights(True)

        for epoch in range(training_steps):
            
            new_obs = model.run_agent(epoch, obs)
            model.evaluate_agent(epoch)

            obs = new_obs

            if epoch %  20000== 0 and len(model.rewards_val_history)> 0 and len(model.rewards_train_history)> 0 and epoch >0:
                print(f"Epoch: {epoch} | Memory size : {len(model.memory)} | Reward eval/Train: {np.mean(model.rewards_val_history)}/{np.mean(model.rewards_train_history)} ")

            if len(model.rewards_val_history)> self.sucess_criteria_epochs :
                if np.mean(model.rewards_val_history[-self.sucess_criteria_epochs:]) >= self.sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break
        
        final_reward = np.mean(model.rewards_train_history[-100:])
        best_epoch_loss = max(best_epoch_loss, final_reward)

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={"total_train_reward" : final_reward})#total_eval_reward

        # Return the evaluation metric value.
        return best_epoch_loss

def run_training(training_steps,   discount,  dense_units_act,  dense_units_crit,num_layer_a,num_layer_c,
                  writer, end_of_episode, save_factor=50000, sucess_criteria_epochs =100, sucess_criteria_value = -100, 
                 environment_name="MountainCar-v0", reward_scaler = 1, evaluation_epoch = 2000,return_agent = False,lr_actor= 0.001, lr_critic_1= 0.001, lr_alpha = 0.001,
                   tau = 0.001, alpha =  0.2, train_epochs = 20, model_path = './checkpoints/SACagent'):
  

    model = SAC(
            discount = discount, 
            dense_units_act = dense_units_act,
            dense_units_crit= dense_units_crit, 
            num_layer_act  = num_layer_a, 
            num_layer_crit= num_layer_c,
            writer = writer,
            trial_n = get_valid_trials_number(writer),
            end_of_episode = end_of_episode,
            evaluation_epoch = evaluation_epoch,
            environment_name = environment_name,
            reward_scaler = reward_scaler,
            lr_actor = lr_actor, lr_critic = lr_critic_1, lr_alpha= lr_alpha, 
            tau= tau,
            train_epochs = train_epochs
            )
    
    obs = model.train_env.reset()
    model.update_weights(True)
    with tqdm.trange(training_steps) as t:
        for epoch in t:
            new_obs = model.run_agent(epoch, obs)
            model.evaluate_agent(epoch)

            obs = new_obs
            
            if epoch %save_factor == 0: 
                model.save_weights(model_path)

            if len(model.rewards_train_history)> 100 :
                if epoch %  model.evaluation_epoch*50 == 0 :
                    print(f"Epoch: {epoch} : Reward Train: {np.mean(model.rewards_train_history[-100:])} ")

                if np.mean(model.rewards_val_history[-sucess_criteria_epochs:]) >= sucess_criteria_value:
                    print("Your agent reached the objetive")
                    break

    if return_agent:
        return model


def final_evaluation(eval_model, eval_env, n_tries=1, exploration ="soft", video_name = "./A3C_soft_video.mp4", sucess_criteria_epochs= 100, reward_scaler = 1, continuous_action_space = False):
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
            
            actions= eval_model.sample_action(obs)
            obs, reward, done, info = eval_env.step(actions)#terminated, truncated , 
            total_reward += reward
            #done = truncated or terminated 

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

