import torch.nn as nn
import torch.nn.functional as F
from modules.agents.obs_encoder import ObsEncoder


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        if args.use_attention:
            self.obs_encoder = ObsEncoder(hidden_dim = 64,map_name = args.env_args['map_name'])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        #input size:(batch_size*parrallel,input_shape) or (batch_size,input_shape) 
        #hidden size:(batch_size*parrallel,rnn_hidden_dim) or (batch_size,rnn_hidden_dim) 
        #x = F.relu(self.fc1(inputs))
        if self.args.use_attention:
            x =self.obs_encoder(inputs)
        else:
            x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
