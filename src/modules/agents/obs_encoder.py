import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ObsEncoder(nn.Module):
    def __init__(self,hidden_dim,map_name = '3m'):
        super(ObsEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        if map_name == '3m':
            self.n_enemies = 3
            self.n_agents = 3
            self.move_feat_dim = 4
            self.enemy_dim = 5
            self.ally_dim = 5 
            self.own_feats_dim = 1
        if map_name == '2s3z':
            self.n_enemies = 5
            self.n_agents = 5
            self.move_feat_dim = 4
            self.enemy_dim = 8
            self.ally_dim = 8
            self.own_feats_dim = 4
        self.action_num = 6 + self.n_enemies
        self.enemy_feats_dim = self.enemy_dim * self.n_enemies
        self.ally_feats_dim = (self.n_agents-1) * self.ally_dim
        self.self_dim = self.move_feat_dim + self.own_feats_dim + self.action_num + self.n_agents
        self.self_encoder = nn.Sequential(
                            nn.Linear(self.self_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),nn.ReLU())
        self.enemy_encoder = nn.Sequential(
                            nn.Linear(self.enemy_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.ally_encoder = nn.Sequential(
                            nn.Linear(self.ally_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.encoder_linear = nn.Sequential(
                            nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh(),nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim))


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        t1 = self.move_feat_dim + self.enemy_feats_dim +self.ally_feats_dim
        t2 = self.action_num + self.n_agents
        self_in = torch.cat((inputs[:, :self.move_feat_dim],inputs[:, t1:t1+self.own_feats_dim],inputs[:,-t2:]),dim=1)
        self_emb = self.self_encoder(self_in)
        #enemy attention
        enemy_emb = []
        for i in range(self.n_enemies):
            enemy_in = inputs[:, self.move_feat_dim+self.enemy_dim*i:self.move_feat_dim+self.enemy_dim*(i+1)]
            enemy_emb.append(self.enemy_encoder(enemy_in))
        enemy_emb = torch.stack(enemy_emb, dim = 2)
        enemy_alpha = F.softmax(torch.matmul(self_emb.view(batch_size,1,-1), enemy_emb)/math.sqrt(self.hidden_dim),dim=2)
        enemy_att = torch.matmul(enemy_alpha,enemy_emb.permute(0,2,1)).squeeze(1)
        enemy_att_out = F.relu(self.layer_norm_1(enemy_att))
        #ally attention
        ally_emb = []
        for i in range(self.n_agents-1):
            ally_in = inputs[:, self.move_feat_dim+self.enemy_feats_dim+self.ally_dim*i:self.move_feat_dim+self.enemy_feats_dim+self.ally_dim*(i+1)]
            ally_emb.append(self.ally_encoder(ally_in))
        ally_emb = torch.stack(ally_emb, dim = 2)
        ally_alpha = F.softmax(torch.matmul(self_emb.view(batch_size,1,-1), ally_emb)/math.sqrt(self.hidden_dim),dim=2)
        ally_att = torch.matmul(ally_alpha,ally_emb.permute(0,2,1)).squeeze(1)
        ally_out = F.relu(self.layer_norm_2(ally_att))
        out = self.encoder_linear(torch.cat([self_emb,enemy_att_out,ally_out],dim=1))
        return out