import torch
import torch.nn as nn
import numpy as np
import random
import os
from model import layers
import scipy.sparse as sp
import copy
from model.Attentions import MultiHeadAttention

def prepare_eg(fp):
    u_features = []
    d_features = []
    u_adjs = []
    d_adjs = []
    for i in range(len(fp)):
        fp_ = fp[i][(fp[i][:, -3] <= 0)] #get back buses
        fp_[:,-2] = fp_[:,-2]/60.0
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            shape=(edges.shape[0], edges.shape[0]))
        # Do not consider ego event in marginal contribution
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj)  # no direction

        u_adjs.append(adj)
        u_features.append(fp_[:, :3 + 1 + 2]) # [occupancy, fh, bh, a, stop_distance, bus_distance]

        fp_ = fp[i][(fp[i][:, -3] >= 0)]
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            shape=(edges.shape[0], edges.shape[0]))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj)  # no direction
        d_adjs.append(adj)
        d_features.append(fp_[:, :3 + 1 + 2])

    return u_adjs, d_adjs, u_features, d_features


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()

    def forward(self, s):
        x = self.elu(self.linear1(s))
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        x = self.elu(self.linear4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, n_stops=22):

        super(Critic, self).__init__()
        self.hidden = 100
        self.state_dim = state_dim

        # for ego critic
        self.fc0 = nn.Linear(state_dim + 1, self.hidden)
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.fc3 = nn.Linear(self.hidden, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(self.hidden, 1)


        self.u_attentions = [
            MultiHeadAttention(d_model = state_dim + 1 + 2, d_proj = self.hidden, num_heads = 1)  for _ in
            range(1)]
        for i, attention in enumerate(self.u_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.u_out_att = MultiHeadAttention(d_model = self.hidden, d_proj = self.hidden, num_heads = 1)

        self.d_attentions = [
            MultiHeadAttention(d_model = state_dim + 1 + 2, d_proj = self.hidden, num_heads = 1)  for _ in
            range(1)]
        for i, attention in enumerate(self.d_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.d_out_att = MultiHeadAttention(d_model = self.hidden, d_proj = self.hidden, num_heads = 1)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.n_stops = n_stops

    def d_egat(self, x, adj):
        x = torch.cat([att(x, x, x, adj) for att in self.d_attentions], dim=1) # x: head x n x 100
        # x = self.d_out_att(x, x, x, adj) # x: nx400
        # x = torch.sum(x, 0) #x:400
        x_target, x_others = x[0, :], torch.sum(x[1:, :], 0)
        return x_target, x_others

    def u_egat(self, x, adj): # x:(nx6)
        x = torch.cat([att(x, x, x, adj) for att in self.u_attentions], dim=1) # x:(nx400)
        # x = self.u_out_att(x, x, x, adj)
        # x = torch.sum(x, 0)
        x_target, x_others = x[0,:], torch.sum(x[1:,:], 0)
        return x_target, x_others

    def event_critic(self, fp):
        u_adjs, d_adjs, u_features, d_features = prepare_eg(fp)
        target = []
        a = []
        reg = []
        for i in range(len(u_adjs)):
            u_x = u_features[i]
            u_adj = u_adjs[i]
            d_x = d_features[i]
            d_adj = d_adjs[i]
            if u_adj.size(0) >= 2:
                u_x_target, u_x_other = self.u_egat(u_x, u_adj)
            else:
                u_x_target, _ = self.u_egat(u_x, u_adj)
                reg.append(torch.square(u_x_target))
                u_x_other = torch.zeros_like(u_x_target)

            if d_adj.size(0) >= 2:
                d_x_target, d_x_other = self.d_egat(d_x, d_adj)
            else:
                d_x_target, _ = self.d_egat(d_x, d_adj)
                reg.append(torch.square(d_x_target))
                d_x_other = torch.zeros_like(d_x_target)
            # u_x = u_x.view(-1, self.hidden) # 1x400
            # d_x = d_x.view(-1, self.hidden)
            # x_target = self.relu(self.fc3(u_x_target + d_x_target))
            # x_other = self.relu(self.fc3(u_x_other + d_x_other))
            # x = u_x + d_x
            # x = torch.cat((u_x,d_x),dim=1)

            # x = self.elu(self.fc3(x))
            x_target = self.elu(self.fc3(u_x_target + d_x_target))
            target.append(self.fc4(x_target))
            x_other = self.elu(self.fc3(u_x_other + d_x_other))
            a.append(self.fc4(x_other))

        a = torch.stack(a, 0).view(-1, 1) # (16x1)
        target = torch.stack(target,0).view(-1, 1)
        if len(reg) > 0:
            reg = torch.stack(reg, 0).view(-1, 1)
        else:
            reg = torch.zeros(1)
        return target, a, reg

    def ego_critic(self, ego):
        out1 = self.fc0(ego)
        out1 = self.relu(out1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        Q = self.fc2(out1)
        return Q

    # x: (16,3)
    # a: (16,1)
    # fp: (16) each of (x,8) x could be 1,2,3...
    def forward(self, xs):
        x, a, fp = xs
        ego = torch.cat([x, a], 1)
        A = self.ego_critic(ego)
        V_target, V_marginal, reg = self.event_critic(fp)
        G = A + V_target + V_marginal
        return A, V_target + V_marginal, G.view(-1, 1), reg


class Agent():
    def __init__(self, state_dim, name, seed=123, n_stops=22, buslist=None):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0

        self.critic = Critic(state_dim, n_stops=n_stops)
        self.critic_target = Critic(state_dim, n_stops=n_stops)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim)
        self.actor_target = Actor(self.state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        a = self.actor(state).squeeze(0).detach().numpy()
        return a

    def learn(self, memories, batch=16):
        if len(memories) < batch:
            return 0, 0

        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp = [], [], [], [], [], []
        memory = random.sample(memories, batch)

        batch_mask = []
        batch_mask_n = []
        batch_fp_critic_t = []
        batch_actor_a = []
        for s, fp, a, r, ns, nfp, in memory:
            batch_s.append(s)
            _fp_ = copy.deepcopy(fp)
            _fp_ = torch.tensor(_fp_, dtype=torch.float32)
            _fp_[0, self.state_dim+1] = self.actor(torch.tensor(s, dtype=torch.float32)).detach()
            # _fp_[0, self.state_dim + 1] = 0
            batch_fp_critic_t.append(_fp_)
            batch_actor_a.append(self.actor(torch.tensor(s, dtype=torch.float32)))
            batch_fp.append(torch.FloatTensor(fp))
            batch_mask.append(len(fp) - 1)
            batch_mask_n.append(len(nfp) - 1)
            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)
            # next line from bonny
            # nfp[0][self.state_dim+1] = self.actor_target(torch.tensor(ns, dtype=torch.float32)).detach()
            batch_nfp.append(torch.FloatTensor(nfp))
        b_fp_pad = batch_fp
        b_nfp_pad = batch_nfp
        batch_actor_a = torch.stack(batch_actor_a, 0)
        b_s = torch.tensor(batch_s, dtype=torch.float)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_ns = torch.tensor(batch_ns, dtype=torch.float)

        def critic_learn():
            Q, A, G, reg = self.critic([b_s, b_a, b_fp_pad])
            Q_, A_, G_, _ = self.critic_target(
                [b_ns, self.actor_target(b_ns).detach(), b_nfp_pad])
            q_target = b_r + self.gamma * (G_.detach()).view(-1, 1)

            loss_fn = nn.MSELoss()
            qloss = loss_fn(G, q_target) + 0.1 * reg.mean()
            self.critic_optim.zero_grad()
            qloss.backward()
            self.critic_optim.step()

            return qloss.item()

        def actor_learn():
            policy_loss, _,  _, _ = self.critic([b_s, batch_actor_a, batch_fp_critic_t])
            policy_loss = -torch.mean(policy_loss)
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            return policy_loss.item()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        qloss = critic_learn()
        policy_loss = actor_learn()

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)
        self.lr_decay()
        self.learn_step_counter += 1

        return policy_loss, qloss

    def lr_decay(self, decay_rate=.99):
        for p in self.critic_optim.param_groups:
            p['lr'] *= decay_rate
        for p in self.actor_optim.param_groups:
            p['lr'] *= decay_rate
    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

    def load(self, model):
        try:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
