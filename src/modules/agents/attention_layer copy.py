import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention_Layer(nn.Module):
    def __init__(self, input_shape, args):
        super(Attention_Layer, self).__init__()
        self.args = args
        self.embedding_dim = 32
        self.hidden_dim =32 
        self.fc1 = nn.Linear(input_shape, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim * 2 ,1)  #self output + attention_output
        #self.weight_psi = nn.Parameter(torch.FloatTensor(self.embedding_dim,self.hidden_dim),requires_grad=True)
        #self.weight_phi = nn.Parameter(torch.FloatTensor(self.hidden_dim,self.embedding_dim),requires_grad=True)
        self.correlation_mat = nn.Parameter(torch.FloatTensor(self.embedding_dim,self.embedding_dim),requires_grad=True)
        #self.weight_psi.data.fill_(0.25)
        #self.weight_phi.data.fill_(0.25)
        self.correlation_mat.data.fill_(0.25)
    '''
    def forward(self, inputs):
        batch_size = self.args.batch_size
        fi = F.relu(self.fc1(inputs))
        fi = fi.view(batch_size,self.args.n_agents,-1)   #(batch_size,n_agents,embedding_dim)
        #fi_T = fi.view(batch_size,-1,self.args.n_agents) #(batch_size,embedding_dim,n_agents)
        fi_T = fi.permute(0,2,1)
        weight = []
        beta_mat = torch.matmul(fi,self.correlation_mat)
        beta_mat = torch.matmul(beta_mat,fi_T)  #(batch_size,n_agents,n_agents)
        for i in range(self.args.n_agents):
            beta_index = list(range(self.args.n_agents))
            beta_index.pop(i)  #delete the agent itself's index
            beta = beta_mat[:,i,beta_index]
            alpha = F.softmax(beta,dim = 1).unsqueeze(2)   ##(batch_size,n_agents-1,1)
            f_j = fi[:,beta_index,:]  #(batch_size,n_agents-1,eb_dim)
            vi = torch.mul(alpha,f_j)
            vi = torch.sum(vi,dim = 1).unsqueeze(1) #(batch_size,1,eb_dim)
            fc2_inputs = torch.cat([fi[:,i,:].unsqueeze(1),vi],dim=2)
            weight_i = F.sigmoid(self.fc2(fc2_inputs))
            weight.append(weight_i)
        weight = torch.stack(weight,dim=1).squeeze(3)
        return weight
    '''

    def forward(self, inputs):
        batch_size = self.args.batch_size
        fi = F.relu(self.fc1(inputs))
        fi = fi.view(batch_size,self.args.n_agents,-1)   #(batch_size,n_agents,embedding_dim)
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
            vi = torch.sum(vi,dim = 1).unsqueeze(1) #(batch_size,1,eb_dim)
            fc2_inputs = torch.cat([fi[:,i].view(batch_size,1,-1),vi],dim=2)
            weight_i = F.sigmoid(self.fc2(fc2_inputs))
            weight.append(weight_i)
        weight = torch.stack(weight,dim=1).squeeze(3)
        return weight
