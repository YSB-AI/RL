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
os.environ["CUDA_VISIBLE_DEVICES"]=""

import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import gym
from collections import deque
import random
import tqdm
import itertools
import shutil
import json
import sklearn
import sklearn.preprocessing


import subprocess
from gym.wrappers.monitoring.video_recorder import VideoRecorder


import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


# Check if GPU is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"GPU is available with {num_gpus} device(s).")
    
    # Get the name of the GPU
    for i in range(num_gpus):
        print(f"GPU {i + 1}: {torch.cuda.get_device_name(i)}")
else:
    print("GPU is not available. Using CPU.")


class ActorCritic(nn.Module):
    def __init__(self, state_shape, n_actions, d, batch_size):
        super(ActorCritic, self).__init__()
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        self.fc0 = nn.Linear(self.state_shape[0], d)
        self.pi = nn.Linear(d, self.n_actions)
        self.v = nn.Linear(d, 1)
        self.distribution = torch.distributions.Categorical

    def forward(self, s):
        s = torch.FloatTensor(s)
        x= self.fc0(s)
        x = torch.relu(x)
        logits = self.pi(x)
        state_value = self.v(x)

        return logits, state_value
    
    def choose_action(self, s, exploration):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=-1).data
        m = self.distribution(prob)
        return m.sample().numpy()
    
    def loss_func(self, s, s_, a, r,d, discount, entropy_factor, debug = False):
     
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        r = torch.tensor(r)
        d = torch.tensor(d)
        a = torch.tensor(a)
        
        if debug:
            print(f"s : {s.shape}")
            print(f"s_ : {s_.shape}")
            #print(f"a : {a.shape}")
            #print(f"r : {r.shape}")
            #print(f"d : {d.shape}")
            
        self.train()
        logits, v = self.forward(s)
        _, next_v= self.forward(s_)
        
        if debug:
            print(f"logits : {logits.shape}")
            print(f"v : {v.shape}")
            print(f"next_v : {next_v.shape}")
            
        v_target = r + discount*next_v*d
        v_target = v_target.detach().requires_grad_() # Stop gradient
        if debug: print(f"v_target : {v_target.shape}")
        
        advantage = v_target - v
        advantage = torch.squeeze(advantage.detach().requires_grad_(), dim =-1) # Stop gradient
        if debug: print(f"advantage : {advantage.shape}")
        
        
        probs = F.softmax(logits, dim=-1)
        if debug: print(f"probs : {probs.shape}")
        
        logprobs = F.log_softmax(logits, dim=-1)
        if debug: print(f"logprobs : {logprobs.shape}")
        
        
        logp_actions = torch.sum(logprobs * F.one_hot(a, self.n_actions), dim=-1) # [n_envs,]
        if debug: print(f"logp_actions : {logp_actions.shape}")
        
        entropy = -torch.sum(probs * logprobs, dim=1)
        if debug: print(f"entropy : {entropy.shape}")
        
        td = v_target - v
        if debug: print(f"td : {td.shape}")
        
        c_loss = 0.5*torch.mean(td.pow(2), dim = 0) # Critic loss
        if debug: print(f"c_loss : {c_loss.shape}")
        
        a_loss =-torch.mean(logp_actions * advantage , dim = 0) -entropy_factor*torch.mean(entropy, dim = 0) # Actor loss
        if debug: print(f"a_loss : {a_loss.shape}")
        
        total_loss = (c_loss + a_loss)
        if debug: print(f"total_loss : {total_loss.shape}")
        return total_loss, c_loss, a_loss
    
        

class A3CAgent_Worker(mp.Process):
    def __init__(self, rank, global_actorcritic,res_queue, lr, entropy_factor, exploration_tech, batch_size,
    discount,  dense_units,  evaluation_epoch = 2500, environment_name="MountainCar-v0" , reward_scaler = 1,
    use_GAE = False, optimizer = None, max_episodes = 2000):
        super(A3CAgent_Worker, self).__init__()
        # Enviroment parameters
        self.environment_name = environment_name
        self.dense_units = dense_units
        self.env = gym.make(environment_name)
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.evaluation_epoch = evaluation_epoch
        self.use_GAE = use_GAE
        self.batch_size = batch_size
        self.process_number = rank
        self.max_episodes = max_episodes
       
        
        # Training parameters
        self.learning_rate= lr
        self.entropy_factor = entropy_factor
        self.exploration_tech = exploration_tech
        self.discount = discount
        self.epsilon = 1
        self.boltzman_factor = 1
        
        self.reward_scaler = reward_scaler

        self.num_processes = 8  # Adjust based on your system capabilities
        self.processes = []

        self.global_actorcritic = global_actorcritic
        self.res_queue = res_queue
        
        self.optimizer=optimizer
        self.train_env =  gym.make(environment_name).env 
        
        self.rewards_train_history = []
        self.rewards_val_history = []
        self.total_rewards = 0

        self.writer = SummaryWriter('Training/torch/worker'+str(self.process_number))


    def evaluate_agent(self, environment_name, episode_counter ):
        env = gym.make(environment_name)  # Implement your environment creation function

        total_reward = 0
        state = env.reset()
        done = False
        
        while not done:
            
            action = self.global_actorcritic.choose_action(state, self.exploration_tech)
            next_state, reward, done, _ = env.step(action)
            total_reward = total_reward + reward
            state = next_state
            
        self.writer.add_scalars(main_tag="Evaluation_Metrics",
                    tag_scalar_dict={"Reward":total_reward},
                    global_step=episode_counter)
                
                

    def run(self):
        env = gym.make(self.environment_name)  # Implement your environment creation function

        local_actorcritic = ActorCritic(
                    state_shape = self.obs_shape,
                    n_actions = self.n_actions, 
                    d = self.dense_units ,
                    batch_size = self.batch_size
                    )
        
        local_actorcritic.load_state_dict(self.global_actorcritic.state_dict())

        
        count_rounds = 0
        total_loss = 0
        
        for episode in range(self.max_episodes):
            #if episode%100 == 0 and self.process_number == 0:
            #    print(f"Process number {self.process_number } - episode {episode}")
            state = env.reset()
            done = False
            
            state_list = []
            next_state_list = []
            action_list = []
            reward_list = []
            done_list = []
            
            
            total_return = 0
            while not done:
                
                action = local_actorcritic.choose_action(state, self.exploration_tech)
                next_state, reward, done, _ = env.step(action)
                
                state_list.append(state)
                next_state_list.append(next_state)
                action_list.append(int(action))
                
                reward_list.append(reward)
                done_list.append(done)

                total_return = total_return+reward
                
                if len(state_list) == self.batch_size:
                    total_loss, c_loss, a_loss = local_actorcritic.loss_func(
                                                            s = np.vstack(state_list),
                                                            s_ = np.vstack(next_state_list),
                                                            a = action_list,
                                                            r = np.array(reward_list).reshape((self.batch_size, 1)),
                                                            d = np.array(done_list).reshape((self.batch_size, 1)),
                                                            discount = self.discount, 
                                                            entropy_factor = self.entropy_factor)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_actorcritic.parameters(), 20.0)  # Gradient clipping
                    for lp, gp in zip(local_actorcritic.parameters(), self.global_actorcritic.parameters()):
                        gp._grad = lp.grad
                    self.optimizer.step()
                    local_actorcritic.load_state_dict(self.global_actorcritic.state_dict())
                    
                    state_list = []
                    next_state_list = []
                    action_list = []
                    reward_list = []
                    done_list = []
                    
                    self.writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"Total":total_loss,
                                            "Actor":a_loss,
                                            "Critic":c_loss},
                                              global_step=episode)
                    
                state = next_state
                
                #self.writer.add_scalar('Actions',torch.tensor(action),count_rounds)
                #self.writer.add_scalar('RT_Rewards',torch.tensor(reward),count_rounds)
                #self.writer.add_scalar('Done',torch.tensor(done),count_rounds)
                
                self.writer.add_scalars(main_tag="Training_Metrics",
                           tag_scalar_dict={"Actions":action,
                                            "RT_Rewards":reward,
                                            "Done":done},
                                              global_step=count_rounds)
                
                count_rounds = count_rounds+1
            

            self.writer.add_scalars(main_tag="Total_rewards",
                           tag_scalar_dict={"episode_reward":total_return},
                                              global_step=count_rounds)
            
            
            
            if episode%100 == 0 and self.process_number == 0: self.evaluate_agent(self.environment_name, episode )
            self.writer.flush()
            
            self.res_queue.put({self.process_number : total_return})
        self.res_queue.put(None)
    
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def run_and_train_agent( learning_rate, entropy_factor, exploration_tech, 
                                discount, dense_units, evaluation_epoch = 500, 
                                environment_name="MountainCar-v0" , reward_scaler = 1, use_GAE = False, 
                                batch_size = 32, tunning_mode = False, max_episodes = 2000, num_processes = 8):

    env = gym.make(environment_name)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    global_actorcritic = ActorCritic(
                    state_shape = obs_shape,
                    n_actions = n_actions, 
                    d = dense_units ,
                    batch_size = batch_size
                    )
    
    global_actorcritic.share_memory()  # Share the model parameters among processes
    #optimizer = SharedAdam(global_actorcritic.parameters(), lr=learning_rate, betas=(0.92, 0.999)) 
    optimizer = optim.Adam(global_actorcritic.parameters(), lr=learning_rate)

      # Adjust based on your system capabilities

    res_queue = mp.Queue()
    results = []
    processes = []

    for rank in range(num_processes):
        p = A3CAgent_Worker(rank, global_actorcritic, res_queue, learning_rate, entropy_factor, exploration_tech, batch_size,
                                discount, dense_units, evaluation_epoch = evaluation_epoch, 
                                environment_name=environment_name, reward_scaler = reward_scaler, use_GAE = use_GAE, 
                                optimizer = optimizer, max_episodes = max_episodes)
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    for rank in range(num_processes): 
        while True:
            r = res_queue.get()
            if r is not None:
                results.append(r)
            else:
                break
            
    results_dict = {}
    for i in range(num_processes):
        results_dict[i] = []

    for l in results:
        k = [key for key in l.keys()][0]
        v = [value for value in l.values()][0]
        results_dict[k].append(v)

    


    if tunning_mode:
        return pd.DataFrame(results_dict)
    else:
        return global_actorcritic, pd.DataFrame(results_dict)

