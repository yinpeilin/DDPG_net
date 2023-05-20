
import json
import math
import sys
sys.path.append("./")
import pdb
import cv2
import os
import time
import os
import game.plane_game as game
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy
import csv




def find_files_with_suffix(folder_path, suffix):
    # 使用os模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)

    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]

    return filtered_files

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(50000)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
 
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
 
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
 
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim1,action_dim2):
        super(Actor, self).__init__()
 
        self.l1 = nn.Linear(state_dim, 20000)
        self.ln1 = nn.LayerNorm(20000)
        self.l1_1 = nn.Linear(20000, 5000)
        self.ln1_1 = nn.LayerNorm(5000)
        self.l2 = nn.Linear(5000, 1280)
        self.ln2 = nn.LayerNorm(1280)
        self.l4 = nn.Linear(1280, 640)
        self.l5 = nn.Linear(1280, 640)
        self.l6 = nn.Linear(640, action_dim1)
        self.l7 = nn.Linear(640, action_dim2)
 
 
 
    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = F.relu(self.ln1_1(self.l1_1(a)))
        a = F.relu(self.ln2(self.l2(a)))
        a1 = F.relu(self.l4(a))
        a2 = F.relu(self.l5(a))
        
        
        a1 = self.l6(a1)
        a2 = self.l7(a2)
        return torch.cat([torch.tanh(a1),a2],1)
 
 
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
 
        self.l1 = nn.Linear(state_dim, 20000)
        self.ln1 = nn.LayerNorm(20000)
        self.l1_2 = nn.Linear(20000, 5000)
        self.ln1_2 = nn.LayerNorm(5000)
        self.l1_3 = nn.Linear(5000, 1000)
        self.ln1_3 = nn.LayerNorm(1000)
        self.l2 = nn.Linear(1000 + action_dim, 5000)
        self.ln2 = nn.LayerNorm(5000)
        self.l2_1 = nn.Linear(5000, 1000)
        self.ln2_1 = nn.LayerNorm(1000)
        self.l2_2 = nn.Linear(1000, 200)
        self.ln2_2 = nn.LayerNorm(200)
        self.l3 = nn.Linear(200, 1)
 
 
    def forward(self, state, action):
        q = F.relu(self.ln1(self.l1(state)))
        q = F.relu(self.ln1_2(self.l1_2(q)))
        q = F.relu(self.ln1_3(self.l1_3(q)))
        q = F.relu(self.ln2(self.l2(torch.cat([q, action], 1))))
        q = F.relu(self.ln2_1(self.l2_1(q)))
        q = F.relu(self.ln2_2(self.l2_2(q)))
        q = self.l3(q)
        
        return q

class DDPG(object):
    def __init__(self, plane_game, state_dim, action_dim, discount=0.95, tau=0.01, learning_rate = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.plane_game = plane_game
        
        self.actor = Actor(state_dim, 2, action_dim-2).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate)
        if len(find_files_with_suffix("./",".pth"))!=0:
            number_list = [int(file_name[0:file_name.find("_")])  for file_name in find_files_with_suffix("./",".pth")]
            number_max = max(number_list)
            print("load the {} model".format(number_max))
            self.load(str(number_max))
        
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
 
        self.discount = discount
        self.tau = tau
        
        self.max_small_epoch = 400
        self.max_test_epoch = 4000
        self.max_save_epoch = 100000
        self.test_num = 10000
        self.epoch = 0
        
        
        self.epoch_range = [20000,400000,600000]
        self.reward_file_path = "./"+str(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))+"reward_change.csv"
 
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # 噪声
        if self.epoch<self.epoch_range[0]:
            action0 = np.array([2*random.random()-1,2*random.random()-1])
            action0 = np.concatenate((action0, np.random.rand(action.shape[0]-2)),axis=0)
            action = action0
            pass
        elif self.epoch<self.epoch_range[1]:
            rate = 0.50
            random.seed(self.epoch)
            action0 = np.array([2*random.random()-1,2*random.random()-1])*rate
            action0 = np.concatenate((action0, np.random.rand(action.shape[0]-2)*rate*0.1),axis=0)
            action += action0
            pass
        elif self.epoch<self.epoch_range[2]:
            rate = 0.20
            action0 = np.array([2*random.random()-1,2*random.random()-1])*rate
            action0 = np.concatenate((action0, np.random.rand(action.shape[0]-2)*rate*0.1),axis=0)
            action += action0
            pass
        else:
            rate = 0.001
            action0 = np.array([2*random.random()-1,2*random.random()-1])*rate
            action0 = np.concatenate((action0, np.random.rand(action.shape[0]-2)*rate*0.1),axis=0)
            action += action0
        
        
        return action
 
    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
 
        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1-done) * self.discount * target_Q.detach()
 
        # Get current Q estimate
        current_Q = self.critic(state, action)
 
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
 
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
 
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.epoch += 1
        if self.epoch % self.max_small_epoch == 0:
            print("已经训练了{}轮，开始更新模型。".format(self.epoch))
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
 
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        if self.epoch % self.max_test_epoch == 0:
            self.test(replay_buffer)
        if self.epoch % self.max_save_epoch == 0:
            self.save(str(self.epoch))
        # Update the frozen target models
    def test(self, ReplayBuffer):
        
        avg_reward = 0.0
        for i in range(self.test_num):
            state,__,__,__,__ = ReplayBuffer.sample(batch_size = 1)
            action = self.select_action(state.cpu().numpy())
            state, reward, done = self.plane_game.frame_step(action)
            avg_reward += reward
        
        
        avg_reward /= self.test_num
        
        
        with open(self.reward_file_path,'a+',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.epoch,avg_reward])
        
        print("现在的网络游戏奖励为{}".format(avg_reward))
        
 
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
 
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
 
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
 
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        # print(self.currentState.shape)

if __name__ == '__main__': 
    # Step 1: init BrainDQN
    
    json_data = json.load(open("assets/data.json"))
    
    
    replay_buffer = ReplayBuffer(json_data["device"]["num_max"]*3+7,json_data["device"]["num_max"]+2)
    
    plane_game = game.GameState(json_data) # Step 3: play game
    
    
    brain = DDPG(plane_game,json_data["device"]["num_max"]*3+7,json_data["device"]["num_max"]+2) # Step 2: init Flappy Bird Game
    # Step 3.1: obtain init state
    state = copy.deepcopy(plane_game.state)
    action0 = np.array([2*random.random()-1,2*random.random()-1])
    action0 = np.concatenate((action0, np.ones(10)),axis=0)
    action0 = np.concatenate((action0, np.zeros(190)),axis=0)
    next_state, reward0, terminal = plane_game.frame_step(action0)

    
    replay_buffer.add(state,action0,next_state,reward0,terminal)
    
    
    # Step 3.2: run the game
    temp = 0
    while True:
        
        
        if temp > 100000:
            plane_game.showMap()
        # plane_game.showMap()
        
        action = brain.select_action(plane_game.state)
        
        
        state = copy.deepcopy(plane_game.state)
        next_state,reward,terminal = plane_game.frame_step(action)
        
        print("reward is {} now".format(reward))
        # 处理nextObservation中的图像
        replay_buffer.add(state,action,next_state,reward,terminal)
        brain.train(replay_buffer,64)
        
        
        temp +=1