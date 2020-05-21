import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop

class AVDLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
        self.mixer = None
        #self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = th.optim.Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # add episode argument to adjust learning rate
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        # Calculate estimated Q-Values
        mac_out = []
        weight_out = []   #save the attention weight
        self.mac.init_hidden(batch.batch_size)   #init agent rnn hidden 
        for t in range(batch.max_seq_length):
            agent_outs,weight_outs = self.mac.forward(batch, t=t, train_weight=True)  
            #agent_outs shape:(batchsize,n_agents,n_actions)   weight_outs shape:(batchsize,n_agents,n_agents)
            mac_out.append(agent_outs)
            weight_out.append(weight_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time  (batch_size,max_seq_length, n_agents, n_actions)
        weight_out = th.stack(weight_out,dim=1)  # Concat over time  (batch_size, max_seq_length, n_agents, 1)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim   (batch_size,max_seq_length-1, n_agents)
        weight_out = weight_out[:,:-1].squeeze(3)
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_weight_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_weight_outs = self.target_mac.forward(batch, t=t, train_weight=True)
            target_mac_out.append(target_agent_outs)
            target_weight_out.append(target_weight_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_weight_out = th.stack(target_weight_out[1:], dim=1).squeeze(3)
        
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
                
        chosen_action_qvals = th.mul(chosen_action_qvals,weight_out)
        chosen_action_qvals = th.sum(chosen_action_qvals,dim = 2).unsqueeze(2)
        target_max_qvals = th.mul(target_max_qvals, target_weight_out)
        target_max_qvals = th.sum(target_max_qvals,dim = 2).unsqueeze(2)
            
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()



        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        #self.adjust_learning_rate(self.optimiser,self.args.lr,t_env) #调整学习率

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def adjust_learning_rate(self, optimizer, lr, t_env, gamma = 0.1):
        if t_env > int(0.5 * self.args.t_max):
            new_lr = lr * gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
