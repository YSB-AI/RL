# %load_ext tensorboard
from datetime import datetime

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

import mlflow

import numpy as np
import gym
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tqdm
import keras_tuner as kt
import tensorflow_probability as tfp
import tf_agents

from keras import backend as K


import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder
#tf.config.run_functions_eagerly(True)


print("Num devices available: ", tf.config.experimental.list_physical_devices())
tf.config.experimental.set_visible_devices([], 'GPU')
tf.debugging.set_log_device_placement(False)

#mlflow_url = "http://127.0.0.1:8080"
#mlflow.set_tracking_uri(uri=mlflow_url)
#mlflow.set_tracking_uri("/media/n/New Volume1/mlflow_data/")
seed =0
np.random.seed(seed)
tf.random.set_seed(seed) #https://github.com/tensorflow/tensorflow/issues/37252

from pympler.tracker import SummaryTracker

class PPO(tf.keras.Model):
    def __init__(self,  discount, dense_units_act_crit,  dense_units_model, num_layer_act_crit, num_layer_model, writer,  lr_actor_critic, lr_model, trial_n = "", 
    evaluation_epoch = 2500, environment_name="",  gae_lambda=0.95, policy_clip = 0.2, training_epoch=20, entropy_coeff = 0.05, memory_size = 50, scaling_factor_reward = 0.1,
    normalize_reward = False, normalize_advantage = False, kl_divergence_target  = 0.01, training_steps = 1000000, sucess_criteria_epochs = 100, sucess_criteria_value = None, 
    use_mlflow = False, reward_norm_factor = 1):
        super(PPO, self).__init__()
        

        self.training_steps = training_steps
        self.sucess_criteria_epochs  = sucess_criteria_epochs 
        self.sucess_criteria_value = sucess_criteria_value
        self.use_mlflow = use_mlflow

        # Enviroment parameters
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.evaluation_epoch = evaluation_epoch
        self.environment_name= environment_name
        self.memory_size = memory_size
        self.reward_norm_factor = reward_norm_factor
        
        self.memory = deque(maxlen= self.memory_size)
        self.episode_reward = 0 
        self.training_epoch = training_epoch
        self.training_counter = 0 
        self.exploration_update = 5000
        self.target_kl = kl_divergence_target
        
        # Training parameters
        
        self.scaling_factor  = scaling_factor_reward
        self.policy_clip = policy_clip
        self.entropy_coeff = entropy_coeff
        
        self.gae_lambda=gae_lambda
        self.discount = discount
        self.normalize_reward = normalize_reward
        self.normalize_advantage = normalize_advantage
        
        
        # Optimizers and loss functions        
        self.actor_critic_optimizer=Adam(learning_rate=lr_actor_critic)
        
        if lr_model is not None and  len(dense_units_model) > 0: 
            self.env_model_optimizer = Adam(learning_rate=lr_model)
            self.env_model_loss_fn = tf.keras.losses.MeanSquaredError()
            self.env_model = EnvironmentModel(env = self.env, obs_shape=self.obs_shape[0],  d = dense_units_model,  num_layer_model = num_layer_model)
        
        # Agents components
        self.actor_critic = Actor_Critic(env = self.env,  d = dense_units_act_crit, num_layer_act_crit = num_layer_act_crit)
        
        

        self.train_env =  gym.make(environment_name)
        
        # Logs parameters
        self.trial_n = trial_n
        self.log_dir = writer + self.trial_n+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        
        self.profiler_log_dir = writer + self.trial_n+ "_profiler_"+ datetime.now().strftime("%Y%m%d-%H%M%S")
       
        if self.use_mlflow : mlflow.set_experiment(f'[{datetime.now().strftime("%Y%m%d-%H%M%S")}] RL Project- PPO for {environment_name} ')
        
        
        self.parameters_to_monitor = {
            "discount" : discount,
            "gae_lambda" : "gae_lambda",
            "environment_name":self.environment_name,
            "lr_actor_critic" : lr_actor_critic,
            "entropy_coeff" : entropy_coeff,
            "policy_clip" : policy_clip,
            "scaling_factor_reward" : scaling_factor_reward,
            "kl_divergence_target" : kl_divergence_target,
            "memory_size" : memory_size,
            "training_epoch" : training_epoch,
            "evaluation_epoch" : evaluation_epoch,
            "normalize_reward": normalize_reward,
            "normalize_advantage" : normalize_advantage
        }
        
        
    
        with self.tb_summary_writer.as_default():
            tf.summary.scalar('discount', discount, step=0)
            tf.summary.scalar('gae_lambda', gae_lambda, step=0)
            for i in range(num_layer_act_crit):
                tf.summary.scalar('dense_units_actor_critic_'+str(i), dense_units_act_crit[i], step=0)
                self.parameters_to_monitor['dense_units_actor_critic_'+str(i)] = dense_units_act_crit[i]

            tf.summary.text('environment_name', self.environment_name , step=0)
            tf.summary.scalar('lr_actor_critic', lr_actor_critic , step=0)
            tf.summary.scalar('entropy_coeff', entropy_coeff, step=0)
            tf.summary.scalar('policy_clip', policy_clip, step=0)
            tf.summary.scalar('scaling_factor_reward', scaling_factor_reward, step=0)
            tf.summary.scalar('kl_divergence_target', kl_divergence_target, step=0)
            tf.summary.scalar('memory_size', memory_size, step=0)
            tf.summary.scalar('training_epoch', training_epoch, step=0)
            tf.summary.scalar('evaluation_epoch', evaluation_epoch, step=0)
            tf.summary.scalar('normalize_reward', int(normalize_reward), step=0)
            tf.summary.scalar('normalize_advantage', int(normalize_advantage), step=0)
            
        
            
        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewars = 0


    def get_sample(self,next_state_value, epoch):
        s, s_, a, policy, v, rewards, dones = [], [], [], [], [], [], []
        
        #[obs, next_obs, action, log_probs, state_value, reward, done,])
        for each in self.memory: 

            s.append(each[0].reshape((1,self.obs_shape[0])))
            s_.append(each[1].reshape((1,self.obs_shape[0])))
            a.append(each[2])
            policy.append(each[3])
            v.append(each[4])
            rewards.append(each[5]/ self.reward_norm_factor)
            dones.append( 1 - int(each[6]))
        
        if self.normalize_reward : 
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)+ 1e-7
            normalized_rewards = [(r - mean_reward) / std_reward for r in rewards]
            rewards = normalized_rewards
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Reward_mean_norm', mean_reward, step=epoch)
                tf.summary.scalar('Reward_std_norm', std_reward, step=epoch)
        
        with self.tb_summary_writer.as_default(): 
            tf.summary.histogram(f'Rewards_for_training', rewards, step=epoch )
        
        v.append(next_state_value)
        
        s = np.array(s).reshape((len(self.memory), self.obs_shape[0]))
        s_ = np.array(s_).reshape((len(self.memory), self.obs_shape[0]))
        
        if self.env.action_space.shape[0] > 1:
            a = np.array(a)
        else:
            a = np.array(a).reshape((len(a)))

        a =  tf.cast(tf.convert_to_tensor(a), dtype = tf.float32)
        v =  tf.cast(tf.convert_to_tensor(v), dtype = tf.float32)
        policy=  tf.cast(tf.convert_to_tensor(policy), dtype = tf.float32)
        rewards= tf.cast(tf.convert_to_tensor(rewards), dtype = tf.float32)
        dones = tf.cast(tf.convert_to_tensor(dones), dtype = tf.float32)

        return s, s_, a, policy, v,rewards,dones
    
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def process_training(self,obs, next_obs, actions, returns, advantages, old_log_policy):
        
            
        with tf.GradientTape() as tape:
            
            # Update actor network
            policy, mu, sigma, log_std, state_value  = self.actor_critic(obs)#, training=True
            
            
            log_policy, entropy = self.actor_critic.logp_and_entropy(mu, sigma, actions)
            log_policy = tf.reshape(log_policy, (len(actions),-1))
            prob_ratio = tf.math.exp(log_policy - old_log_policy) 
            
            weighted_clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
            #Update critic network
            # print("obs ->",obs.shape)
            # print("next_obs ->",next_obs.shape)
            # print("actions ->",actions.shape)
            # print("log_policy ->",log_policy.shape)
            # print("old_log_policy ->",old_log_policy.shape)
            # print("policy ->",policy.shape)
            
            # print("advantages ->",advantages.shape)
            # print("prob_ratio ->",prob_ratio.shape)
            # print("weighted_clipped_probs ->",weighted_clipped_probs.shape)
            
            # print("state_value ->",state_value.shape)
            # print("returns ->",returns.shape)
            
            #Entropy regularization
            entropy_loss = -tf.reduce_mean(entropy)
            # print("entropy ->",entropy.shape)
            
            #clipped_surr_actor_loss = -tf.reduce_mean(tf.math.minimum(prob_ratio * adv, min_adv)) 
            
            critic_loss =tf.reduce_mean((state_value - returns)**2, axis=0) *0.5
            clipped_surr_actor_loss = -tf.reduce_mean(tf.minimum(prob_ratio * advantages, weighted_clipped_probs * advantages)) 
            
            actor_loss = clipped_surr_actor_loss +  self.entropy_coeff * entropy_loss
            total_loss = actor_loss + critic_loss 
            
            approx_kl_divergence = tf.reduce_mean(old_log_policy - log_policy)                                     # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
            kl_penalty = tf.maximum(0.0, approx_kl_divergence - self.target_kl)
            
            total_loss = total_loss+kl_penalty

            
        actor_critic_grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic_optimizer.apply_gradients(zip(actor_critic_grads, self.actor_critic.trainable_variables))
        
        # print("approx_kl ->",approx_kl.shape)
        # print("critic_loss ->",critic_loss.shape)
        # print("clipped_surr_actor_loss ->",clipped_surr_actor_loss.shape)
        # print("total_loss ->",total_loss.shape)
        
        return total_loss, actor_loss, critic_loss, state_value, entropy, prob_ratio, actor_critic_grads, approx_kl_divergence
    
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def training_env_model(self, obs, next_obs, action_arr):
        
        with tf.GradientTape() as tape:
            predicted_next_obs = self.env_model(obs,tf.reshape(action_arr, (len(action_arr),self.env.action_space.shape[0])),)
            model_loss = self.env_model_loss_fn(next_obs, predicted_next_obs)

        model_grads = tape.gradient(model_loss, self.env_model.trainable_variables)
        self.env_model_optimizer.apply_gradients(zip(model_grads, self.env_model.trainable_variables))
        
        return model_loss, model_grads
    
    
    def train(self, next_state_value, epoch):
        
        obs, next_obs, actions, old_log_policy, state_values, rewards, dones =  self.get_sample(next_state_value, epoch)
        
        final_value = state_values[-1] 
        state_values = state_values[:-1] 
        
        
        rewards = tf.cast(tf.reshape(rewards, (len(rewards),1)), dtype = tf.float32)
        final_value = tf.cast(tf.reshape(final_value, (1,)), dtype = tf.float32)
        state_values = tf.cast(tf.reshape(state_values, (len(state_values),1)), dtype = tf.float32)

        discounts = self.discount *  dones
        discounts = tf.cast(np.array(discounts).reshape((len(discounts),1)), dtype = tf.float32)

        advantages = tf_agents.utils.value_ops.generalized_advantage_estimation(
            values = state_values, final_value = final_value, discounts =  discounts, 
            rewards = rewards, td_lambda=self.gae_lambda, time_major=True
        )
        
        
        returns=  tf_agents.utils.value_ops.discounted_return(rewards,discounts,time_major=True, final_value=final_value, provide_all_returns = True) 
        returns = tf.squeeze(returns, axis = -1)

        if self.normalize_advantage :
            advantage_mean =  tf.squeeze(tf.reduce_mean(advantages, axis=0), axis = 0)
            advantage_std =  tf.squeeze(tf.math.reduce_std(advantages, axis=0), axis = 0) + 1e-7
            advantages = (advantages - advantage_mean)/advantage_std
            advantages= tf.cast(tf.convert_to_tensor(advantages), dtype = tf.float32)
            
            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Advantage_mean_norm', advantage_mean, step=epoch)
                tf.summary.scalar('Advantage_std_norm', advantage_std, step=epoch)
        
        
        obs = tf.cast(tf.convert_to_tensor(obs), dtype = tf.float32)
        next_obs  = tf.cast(tf.convert_to_tensor(next_obs), dtype = tf.float32)
        old_log_policy = tf.cast(tf.convert_to_tensor(old_log_policy), dtype = tf.float32)
        
        #for k in range(self.training_epoch):
        total_loss, actor_loss, critic_loss, state_value, entropy, prob_ratio, actor_critic_grads, approx_kl = self.process_training(obs, next_obs, actions, returns, advantages, old_log_policy)
        model_loss, model_grads = self.training_env_model(obs, next_obs, actions)
        
        # if approx_kl > self.target_kl: #1.5 *
        #     print(f"Early stopping at step {k} due to reaching max kl.")
        #     break
        
        
        
        with self.tb_summary_writer.as_default():
            for i, grad in enumerate(actor_critic_grads):
                tf.summary.histogram(f'ActorCritic/Gradients/Layer_{i}', grad, step=self.training_counter )
                
            
            if model_grads is not None:
                for i, grad in enumerate(model_grads):
                    tf.summary.histogram(f'Env/Gradients/Layer_{i}', grad, step=self.training_counter )
                
   
        self.training_counter  = self.training_counter  +1
        
        return total_loss, actor_loss, critic_loss, advantages, state_value, returns, entropy, prob_ratio, model_loss, approx_kl
  
    
    def calculate_intrinsic_reward(self, obs, action, next_obs):
        predicted_next_obs = self.env_model(obs, action)
        predicted_next_obs = predicted_next_obs.numpy()
        prediction_error = np.mean(np.square(predicted_next_obs - next_obs))

        return prediction_error, predicted_next_obs[0]


    def run_agent(self, epoch, obs):
        action, mu, sigma, log_probs, state_value = self.actor_critic(obs.reshape((1,self.obs_shape[0])))
        if self.environment_name in ["BipedalWalker-v3", "Ant-v2"]:
            if not isinstance(action, float) :
                action = action[0,]
        #print(action)
        next_obs, reward, done, info = self.train_env.step(action)
        intrinsic_reward, predicted_next_obs = self.calculate_intrinsic_reward( obs.reshape((1,self.obs_shape[0])), # REVIEW 
                                                           tf.reshape(action, shape =(1,self.env.action_space.shape[0]) ),
                                                           next_obs.reshape((1,self.obs_shape[0])))
        
        total_reward = reward + self.scaling_factor * intrinsic_reward
        #done = truncated or terminated  #terminated, truncated,
        self.episode_reward += reward

        
        self.memory.append([obs, next_obs, action, log_probs,  state_value, total_reward, done])
        self.append_interaction_metrics(action, obs, reward,total_reward, done, epoch)
        
        if done or len(self.memory)== self.memory_size:        
            
            _,_,_,_,next_state_value = self.actor_critic(next_obs.reshape((1,self.obs_shape[0])))
            
            total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, model_loss, approx_kl = self.train(next_state_value, epoch)          
            
            self.memory.clear()
                
            if done :
                self.append_training_metrics(self.episode_reward, total_loss, actor_loss, critic_loss, advantages, state_values, returns, 
                                             entropy, prob_ratio, model_loss, approx_kl, epoch)
                self.rewards_train_history.append(self.episode_reward)
                self.episode_reward = 0 
                next_obs = self.train_env.reset()
                #predicted_next_obs = next_obs

        return next_obs#predicted_next_obs#next_obs#predicted_next_obs
        

    def evaluate(self, eval_env, n_tries=1, hyp= False):
        rewards_history = []
        for _ in range(n_tries):
            obs = eval_env.reset()
            total_reward = 0
            while(True):
                
                actions, _, _, _ ,_ = self.actor_critic(obs.reshape((1,self.obs_shape[0])))
                
                if self.environment_name in ["BipedalWalker-v3", "Ant-v2"]:
                    if not isinstance(actions, float) :
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
            rewards_history = np.mean(self.evaluate(gym.make(self.environment_name) , n_tries=1))
            self.rewards_val_history.append(rewards_history)

            with self.tb_summary_writer.as_default():
                tf.summary.scalar('Eval_rewards', self.rewards_val_history[-1], step=epoch)

                if len(self.rewards_val_history)> sucess_criteria_epochs: 
                    eval_epoch = len(self.rewards_val_history)
                    tf.summary.scalar('Eval_average_rewards', np.mean(self.rewards_val_history[-sucess_criteria_epochs:]), step=eval_epoch)
                    tf.summary.scalar('Train_average_rewards', np.mean(self.rewards_train_history[-sucess_criteria_epochs:]), step=epoch)     
        
        
    def append_interaction_metrics(self, action, obs, reward,  total_reward, done, epoch):

        with self.tb_summary_writer.as_default():
            tf.summary.histogram('Action', action, step=epoch)
            tf.summary.histogram('Observation', obs, step=epoch)
            
            for i in range(self.env.observation_space.shape[0]):
                tf.summary.scalar('Observation component '+str(i), obs[i], step=epoch)
                
            for i in range(self.env.action_space.shape[0]):
                tf.summary.scalar('Action component '+str(i), action[i], step=epoch) 
                
                
            tf.summary.scalar('RT_Reward', reward, step=epoch)
            tf.summary.scalar('Intrinsic_Reward', total_reward, step=epoch)
            
            if done == 1:
                tf.summary.scalar('Done', 1, step=epoch)
                tf.summary.scalar('Done', 0, step=epoch+1)
            
        if self.use_mlflow:    
            for i in range(self.env.observation_space.shape[0]):
                mlflow.log_metric('Observation component '+str(i), obs[i], step=epoch)
                
            for i in range(self.env.action_space.shape[0]):
                mlflow.log_metric('Action component '+str(i), action[i], step=epoch) 
                
                
            mlflow.log_metric('RT_Reward', reward, step=epoch)
            mlflow.log_metric('Intrinsic_Reward', total_reward, step=epoch)
            mlflow.log_metric('Done', done, step=epoch)
        

    def append_training_metrics(self, episode_reward, total_loss, actor_loss, critic_loss, advantages, state_values, returns, entropy, prob_ratio, model_loss, approx_kl, epoch):
       
        with self.tb_summary_writer.as_default():
            tf.summary.scalar('Episode_Reward', episode_reward, step=epoch)
            tf.summary.scalar('Entropy', tf.reduce_mean((entropy)), step=epoch)
            tf.summary.scalar('KL Divergence', approx_kl, step=epoch)
            tf.summary.scalar('total_loss', tf.reduce_mean((total_loss)), step=epoch)
            tf.summary.scalar('actor_loss', tf.reduce_mean((actor_loss)), step=epoch)
            tf.summary.scalar('critic_loss', tf.reduce_mean((critic_loss)), step=epoch)
            if model_loss != None : tf.summary.scalar('model_loss', model_loss, step=epoch)
            tf.summary.scalar('Advantages', tf.reduce_mean((advantages)), step=epoch)
            tf.summary.scalar('State_Value', tf.reduce_mean(tf.convert_to_tensor(state_values)), step=epoch)
            tf.summary.scalar('Expected_Return', tf.reduce_mean(tf.convert_to_tensor(returns)), step=epoch)
            tf.summary.scalar('Prob_ratio', tf.reduce_mean(tf.convert_to_tensor(prob_ratio)), step=epoch)
            
        if self.use_mlflow:  
            mlflow.log_metric('Episode_Reward', episode_reward, step=epoch)
            mlflow.log_metric('Entropy', tf.reduce_mean((entropy)), step=epoch)
            mlflow.log_metric('KL Divergence', approx_kl, step=epoch)
            mlflow.log_metric('total_loss', tf.reduce_mean((total_loss)), step=epoch)
            mlflow.log_metric('actor_loss', tf.reduce_mean((actor_loss)), step=epoch)
            mlflow.log_metric('critic_loss', tf.reduce_mean((critic_loss)), step=epoch)
            if model_loss != None : mlflow.log_metric('model_loss', model_loss, step=epoch)
            mlflow.log_metric('Advantages', tf.reduce_mean((advantages)), step=epoch)
            mlflow.log_metric('State_Value', tf.reduce_mean(tf.convert_to_tensor(state_values)), step=epoch)
            mlflow.log_metric('Expected_Return', tf.reduce_mean(tf.convert_to_tensor(returns)), step=epoch)
            mlflow.log_metric('Prob_ratio', tf.reduce_mean(tf.convert_to_tensor(prob_ratio)), step=epoch)
            


    def train_agent(self):
        if self.sucess_criteria_value is None :
            raise Exception ("Missing sucess criteria !")
        
        if self.use_mlflow: 
            with mlflow.start_run():
                mlflow.log_params(self.parameters_to_monitor)
                
                obs = self.train_env.reset()
                for epoch in range(self.training_steps):
                    next_obs = self.run_agent(epoch, obs)
                    self.evaluate_agent(epoch)

                    obs = next_obs
                    
                    if epoch %  2000 == 0 and len(self.rewards_val_history)> 0 and len(self.rewards_train_history)> 0 and epoch >0:
                        print(f"Epoch: {epoch} : Reward eval/Train: {np.mean(self.rewards_val_history)}/{np.mean(self.rewards_train_history)} ")

                    if len(self.rewards_val_history)> self.sucess_criteria_epochs :
                        if np.mean(self.rewards_val_history[-self.sucess_criteria_epochs:]) >= self.sucess_criteria_value:
                            print("Your agent reached the objetive")
                            break
        else : 
            obs = self.train_env.reset()
            #tracker = SummaryTracker()
            #tf.profiler.experimental.start(self.profiler_log_dir)
     

            for epoch in range(self.training_steps):
                #with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):

                    #with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):
                    next_obs = self.run_agent(epoch, obs)
                    self.evaluate_agent(epoch)
                

                    obs = next_obs
                    if epoch %  2000 == 0 and len(self.rewards_val_history)> 0 and len(self.rewards_train_history)> 0 and epoch >0:
                        print(f"Epoch: {epoch} : Reward eval/Train: {np.mean(self.rewards_val_history)}/{np.mean(self.rewards_train_history)} ")

                    if len(self.rewards_val_history)> self.sucess_criteria_epochs :
                        if np.mean(self.rewards_val_history[-self.sucess_criteria_epochs:]) >= self.sucess_criteria_value:
                            print("Your agent reached the objetive")
                            break
                #tracker.print_diff()
            #tf.profiler.experimental.stop() 
  

        return  epoch
                
class Actor_Critic(tf.keras.Model):
    def __init__(self,  env,  d, num_layer_act_crit = 1):
        super(Actor_Critic, self).__init__()
        self.env = env
        self.num_layers = num_layer_act_crit
        self.dense_layers = [Dense(d[i], activation='relu', name ="inp_"+str(i)) for i in range(num_layer_act_crit)]
        self.d_state_value = Dense(1,  name ="v")

        self.noise= 1e-7
        self.mean_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.tanh, name='actor_mu' )
        self.sigma_dense = Dense(units=self.env.action_space.shape[0], activation=tf.nn.softplus, name='actor_sigma' )
        
        
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def call(self, observations):

        # Get mean and standard deviation from the policy network
        x = observations
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
        
        mu = self.mean_dense(x)
        sigma = self.sigma_dense(x)
        mu, sigma = tf.squeeze(mu*self.env.action_space.high[0]), tf.squeeze(sigma + self.noise)
        state_value = self.d_state_value(x)
        
        normal_dist = tfp.distributions.Normal(mu, sigma)
        action = tf.clip_by_value(normal_dist.sample(1) , self.env.action_space.low[0], self.env.action_space.high[0])
        log_std = normal_dist.log_prob(action)
        
        try:
            action = tf.reshape(action, (len(observations), -1))
            if action.shape[1] >= 1:
                action = tf.squeeze(action, axis = -1)
                
        except :
            pass
        return action, mu, sigma, log_std, tf.squeeze(state_value, axis = -1)
    
    @tf.function( experimental_relax_shapes=True, reduce_retracing=True)
    def logp_and_entropy(self, mu, sigma, action):
        normal_dist = tfp.distributions.Normal(mu, sigma)
        log_std = normal_dist.log_prob(action)
        entropy = normal_dist.entropy() 
        return log_std, entropy

    
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
                  gae_factor = None,
                  gae_min = 0.90, gae_max = 0.99,
                  lr_actor_crit_min = 0.00001, lr_actor_crit_max = 0.005,
                  lr_model_min = None, lr_model_max = None,
                  dense_min = 32, dense_max = 200,
                  environment_name="MountainCar-v0" , 
                  dense_layers = None,
                  num_layers_act = None, max_num_layers_act = 1, kl_divergence_target = None, num_layers_model = None, 
                  training_epoch =50, entropy_factor_max = 0.1, entropy_factor_min = 0.0001,
                  entropy_factor = None,
                  memory_size = 50, training_epoch_max = None, memory_size_max = None, normalize_reward = False, normalize_advantage = False, 
                  scaling_factor_reward = None, use_mlflow = False,
                  reward_norm_factor  = 1):
        
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
        
        self.lr_actor_crit_min = lr_actor_crit_min
        self.lr_actor_crit_max = lr_actor_crit_max
        
       
        self.lr_model_min = lr_model_min
        self.lr_model_max = lr_model_max
        self.dense_layers = dense_layers

        self.dense_min = dense_min
        self.dense_max = dense_max
        self.environment_name  = environment_name


        self.num_layers_act = num_layers_act
        self.max_num_layers_act = max_num_layers_act
        
        self.gae_factor = gae_factor
        self.gae_min = gae_min
        self.gae_max = gae_max
        self.training_epoch = training_epoch
        self.memory_size = memory_size

        self.entropy_factor_min = entropy_factor_min
        self.entropy_factor_max = entropy_factor_max
        self.entropy_factor = entropy_factor
        
        self.num_layers_model  = num_layers_model
        
        self.discount = discount
        
        self.policy_clip = policy_clip
        self.normalize_rewards = normalize_reward
        self.normalize_advantages = normalize_advantage
        
        self.scaling_factor_reward = scaling_factor_reward
        
        self.kl_divergence_target  = kl_divergence_target
        self.use_mlflow = use_mlflow
        self.reward_norm_factor=reward_norm_factor

    def build(self, hp):
        K.clear_session()

        if self.training_epoch_max != None:
            self.training_epoch = hp.Int('training_epoch', 10, self.training_epoch_max, step = 10)
            
        if self.memory_size_max != None:
            self.memory_size = hp.Int('memory_size_max', 10, self.memory_size_max, step = 8)
        
        
        discount = self.discount
        if discount is None: discount =  hp.Float('discount',self.discount_min, self.discount_max, step =0.01)
        
        if self.gae_factor is None:
            gae_lambda =  hp.Float('gae_lambda',self.gae_min, self.gae_max, step =0.01)
        else:
            gae_lambda = self.gae_factor
            
        
        lr_actor_critic = hp.Float('lr_actor_critic', self.lr_actor_crit_min, self.lr_actor_crit_max)
        if self.lr_model_max is not None and self.num_layers_model is not None:
            lr_model = hp.Float('lr_model', self.lr_model_min, self.lr_model_max)
        
        policy_clip = self.policy_clip
        if policy_clip is None: policy_clip = hp.Float('policy_clip', 0.1, 0.3, step =0.1)
        
        scaling_factor_reward = self.scaling_factor_reward 
        if self.scaling_factor_reward == None:
            scaling_factor_reward = hp.Float('scaling_factor_reward', 0.01, 0.3)
            
        entropy_coeff = self.entropy_factor
        if self.entropy_factor is None :entropy_coeff = hp.Float('entropy_coeff', self.entropy_factor_min, self.entropy_factor_max)

        if self.dense_layers is None : 
        
            num_layer_act_crit = self.num_layers_act
            if self.num_layers_act is None:
                if self.max_num_layers_act>1 :
                    num_layer_act_crit = hp.Int('n_dense_layers_actor', 1, self.max_num_layers_act)
                else : 
                    num_layer_act_crit = self.max_num_layers_act
                
            dense_units_act_crit =  [hp.Int('dense_units_act_crit_'+str(i), self.dense_min, self.dense_max) for i in range(num_layer_act_crit)]
        else : 
            
            dense_units_act_crit = self.dense_layers
            num_layer_act_crit = len(dense_units_act_crit)
        
        
        dense_units_model = []
        num_layer_m = self.num_layers_model
        if self.lr_model_max is not None and self.num_layers_model is not None:
            if self.num_layers_model>1 :
                num_layer_m = hp.Int('n_dense_layers_model', 1, self.num_layers_model)
                
            dense_units_model =  [hp.Int('n_dense_layers_model'+str(i), self.dense_min, self.dense_max) for i in range(num_layer_m)]
        
        if self.kl_divergence_target is None:
            kl_divergence_target = hp.Float('kl_divergence_target', 0.01, 0.2)
        else:
            kl_divergence_target = self.kl_divergence_target

        actorcritic_agent = PPO(
            training_steps = self.training_steps, 
            sucess_criteria_epochs  = self.sucess_criteria_epochs, 
            sucess_criteria_value = self.sucess_criteria_value,
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
            normalize_reward = self.normalize_rewards,
            normalize_advantage = self.normalize_advantages, 
            kl_divergence_target = kl_divergence_target,
            use_mlflow = self.use_mlflow,
            reward_norm_factor = self.reward_norm_factor
            )
        return actorcritic_agent

    def fit(self, hp, model, x, y,  callbacks=None, **kwargs):
        
        
        # Record the best validation loss value
        best_epoch_loss = float(-100000)
        
        # Assign the model to the callbacks.rewards_val_history
        for callback in callbacks:
            callback.model = model
        
        last_epoch = model.train_agent()
        
        final_reward = np.mean(model.rewards_train_history)
        best_epoch_loss = max(best_epoch_loss, final_reward)

        for callback in callbacks:
            # The "my_metric" is the objective passed to the tuner.
            callback.on_epoch_end(last_epoch, logs={"total_train_reward" : final_reward})#total_eval_reward

        # Return the evaluation metric value.
        return best_epoch_loss


def rerun_training(training_steps, model):
    
    model.training_steps = training_steps
    last_epoch = model.train_agent()

    return model

def run_training(training_steps,   discount,  dense_units_act_crit,  dense_units_model,num_layer_a_c,num_layer_m,
                  writer,  save_factor=50000, sucess_criteria_epochs =100, sucess_criteria_value = -100, 
                 environment_name="MountainCar-v0",  evaluation_epoch = 2000,return_agent = False,
                 lr_actor_critic= 0.001,  lr_model= 0.00001,  gae_lambda=0.95, training_epoch= 20, entropy_coeff= 0.01, 
                 policy_clip = 0.2,memory_size= 50, id = 1, normalize_reward = False, normalize_advantage = False,  scaling_factor_reward = 0.1,
                 kl_divergence_target = 0.01, use_mlflow = False, reward_norm_factor = 1):
  
    model = PPO(
        training_steps = training_steps, 
        sucess_criteria_epochs  = sucess_criteria_epochs, 
        sucess_criteria_value = sucess_criteria_value,
        discount = discount, 
        dense_units_act_crit = dense_units_act_crit,
        dense_units_model = dense_units_model, 
        num_layer_act_crit  = num_layer_a_c, 
        num_layer_model = num_layer_m,
        writer = writer,
        trial_n = get_valid_trials_number(writer ),
        evaluation_epoch = evaluation_epoch,
        environment_name = environment_name,
        lr_actor_critic = lr_actor_critic,lr_model=lr_model,
        gae_lambda=gae_lambda,
        policy_clip = policy_clip,
        training_epoch = training_epoch ,
        entropy_coeff = entropy_coeff,
        memory_size = memory_size,
        normalize_reward = normalize_reward,
        normalize_advantage = normalize_advantage,
        scaling_factor_reward=scaling_factor_reward,
        kl_divergence_target = kl_divergence_target,
        use_mlflow = use_mlflow,
        reward_norm_factor = reward_norm_factor
    )
    
    model.train_agent()

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
            actions, _, _, _ ,_ = eval_model.actor_critic(obs.reshape((1,eval_model.obs_shape[0])))
                     
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
