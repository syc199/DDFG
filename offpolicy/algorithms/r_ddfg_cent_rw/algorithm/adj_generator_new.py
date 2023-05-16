#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:03:55 2018

@author: xinruyue
"""
import sys
sys.path.append('..')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from offpolicy.utils.util import gumbel_softmax_mdfg, to_torch, update_linear_schedule, DecayThenFlatSchedule
from offpolicy.algorithms.utils.autoencoder import Autoencoder
from offpolicy.algorithms.utils.adj_policy import AdjPolicy
# Network Generator
# 此类为一个利用Gumbel softmax生成离散网络的类
class Adj_Generator(nn.Module):
    def __init__(self, args, obs_dim ,state_dim, device):
        super(Adj_Generator, self).__init__()
        self.adj_hidden_dim = args.adj_hidden_dim
        self.adj_output_dim = args.adj_output_dim

        self.num_variable = args.num_agents
        self.num_factor = args.num_factor
        self.alpha = args.adj_alpha
        self.num = 1
        self.device = device
        self.autoencoder = Autoencoder(args.hidden_size,self.adj_hidden_dim,self.adj_output_dim,args.use_orthogonal,args.use_ReLU,device)
        self.adj_policy = AdjPolicy(args,obs_dim ,state_dim,device,args.use_ReLU)
        self.exploration = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.adj_anneal_time,decay="linear")
        #gen_matrix 为邻接矩阵的概率
        self.device = device
        self.highest_orders = args.highest_orders
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.to(device) 

    def sample(self, obs, state, explore=False, t_env=None):
        # 采样——得到一个临近矩阵   
        #print("0:{}".format(torch.cuda.memory_allocated(0)))
        batch_size = obs.shape[0]
        input_batch = to_torch(obs).to(**self.tpdv)
        if len(state.shape) == 1:
            state_batch = to_torch(state).to(**self.tpdv).unsqueeze(0)
        else:
            state_batch = to_torch(state).to(**self.tpdv)

        embedding, output = self.autoencoder(input_batch)
        stack_exp = self.adj_policy(embedding,state_batch)
        stack_exp_norm = stack_exp / (stack_exp.sum(dim=1,keepdim=True) + 1e-20)
        softmax = torch.clamp(stack_exp_norm,1e-3,1-1e-3)
        softmax_pre =  softmax.transpose(1,2)
        if explore:
            eps = torch.tensor(self.exploration.eval(t_env))
            rand_numbers = torch.rand(batch_size,self.num_factor,1)
            take_random = torch.where(rand_numbers < eps,torch.ones_like(rand_numbers,dtype=torch.int64),torch.zeros_like(rand_numbers,dtype=torch.int64)).to(self.device)
            random_indices = torch.multinomial(softmax_pre.reshape(-1,self.num_variable), self.highest_orders,replacement=False).reshape(batch_size,-1,self.highest_orders)
            greedy_indices = torch.topk(softmax_pre.reshape(-1,self.num_variable), k=self.highest_orders, dim=1, largest=True)[1].reshape(batch_size,-1,self.highest_orders)
            indices =  (1 - take_random) * greedy_indices + take_random * random_indices
        else:
            indices = torch.topk(softmax_pre.reshape(-1,self.num_variable), k=self.highest_orders, dim=1, largest=True)[1].reshape(batch_size,-1,self.highest_orders)
            
        entropy = -softmax * torch.log(softmax)
        x = torch.ones_like(softmax,dtype=torch.int64)
        y = torch.zeros_like(softmax,dtype=torch.int64)
        cond_adj_1 = torch.where(softmax>1/(self.num_variable * self.highest_orders),x,y)
        cond_adj_2 = torch.zeros_like(softmax,dtype=torch.int64)
        cond_adj_2= cond_adj_2.transpose(1,2).scatter(2,indices,1).transpose(1,2)
        cond_adj = cond_adj_1 & cond_adj_2
        prob_adj = torch.log(torch.where(cond_adj==1,softmax,torch.ones_like(softmax,dtype=torch.float32)))

        
        return prob_adj, cond_adj, entropy.sum(-2).mean(-1), softmax

    def parameters(self):
        parameters_sum = []
        parameters_sum += self.autoencoder.parameters()
        parameters_sum += self.adj_policy.parameters()

        return parameters_sum

    def load_state(self, source_adjnetwork):
        self.autoencoder.load_state_dict(source_adjnetwork.autoencoder.state_dict())
        self.adj_policy.load_state_dict(source_adjnetwork.adj_policy.state_dict())
