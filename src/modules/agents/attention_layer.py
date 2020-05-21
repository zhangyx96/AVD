import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.agents.obs_encoder import ObsEncoder


class Attention_Layer(nn.Module):
    def __init__(self, input_shape, args):
        super(Attention_Layer, self).__init__()
        self.args = args
        self.embedding_dim = 64
        self.hidden_dim = 64
        #sself.fc1 = nn.Linear(input_shape, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim * 2 ,1)  #self output + attention_output
        self.correlation_mat = nn.Parameter(torch.FloatTensor(self.embedding_dim,self.embedding_dim),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)
        self.layer_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_dim)
        self.g = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.obs_encoder = ObsEncoder(hidden_dim = self.hidden_dim,map_name = args.env_args['map_name'])
        self.last = nn.Sequential(nn.Linear(self.embedding_dim * 2 ,1), nn.Sigmoid())

    def forward(self, inputs):
        batch_size = self.args.batch_size
        fi = self.obs_encoder(inputs).view(batch_size,self.args.n_agents,-1)  #(batch_size,n_agents,embedding_dim)
        weight = []
        for i in range(self.args.n_agents):
            beta = []
            f_j = []
            for j in range(self.args.n_agents):
                if i!=j:
                    f_j.append(fi[:,j].view(batch_size,1,-1))    #(batch_size,1,eb_dim)
                    beta_i_j = torch.matmul(fi[:,i].view(batch_size,1,-1),self.correlation_mat)
                    beta_i_j = torch.matmul(beta_i_j,fi[:,j].view(batch_size,-1,1))
                    beta.append(beta_i_j.squeeze(1).squeeze(1))
            f_j = torch.stack(f_j,dim = 1).squeeze(2)  #(batch_size,n_agents-1,eb_dim)
            beta = torch.stack(beta,dim = 1)            
            alpha = F.softmax(beta,dim = 1).unsqueeze(2)  #(batch_size,n_agents-1,1)
            vi = torch.mul(alpha,f_j)
            vi = torch.sum(vi,dim = 1) #(batch_size,1,eb_dim)
            vi = F.relu(self.layer_norm_1(vi))
            gi = fi[:,i]
            all_inputs = torch.cat([gi,vi],dim=1)
            weight_i = self.last(all_inputs)
            weight.append(weight_i)
        weight = torch.stack(weight,dim=1)
        return weight
