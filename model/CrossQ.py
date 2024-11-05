import torch
import torch.nn as nn
import random
import os
import torch.nn.functional as F
from model.Attentions import MultiHeadAttention
import numpy as np
from model.RNormalization import BatchRenorm
import scipy.sparse as sp

def prepare_eg(fp):
    u_features = []
    d_features = []
    u_adjs = []
    d_adjs = []
    for i in range(len(fp)):
        fp_ = fp[i][(fp[i][:, -3] <= 0)]  # get back buses
        fp_[:, -2] = fp_[:, -2] / 60.0
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
        u_features.append(fp_[:, :3 + 1 + 2])  # [occupancy, fh, bh, a, stop_distance, bus_distance]

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


class CrossQ_Actor(nn.Module):
    def __init__(self, input_size, hidden_size=2048, output_size=1, use_batch_norm=True, use_renorm=False, initial_dropout_rate=0.03, decay_rate=0.995):
        super(CrossQ_Actor, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_renorm = use_renorm
        self.initial_dropout_rate = initial_dropout_rate
        self.dropout_rate = initial_dropout_rate  # start with 3% dropout
        self.decay_rate = decay_rate  # dropout decay rate (0.995)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(hidden_size, momentum=0.99)
            self.batch_norm2 = nn.BatchNorm1d(hidden_size, momentum=0.99)
        elif use_renorm:
            self.batch_norm1 = BatchRenorm(hidden_size)
            self.batch_norm2 = BatchRenorm(hidden_size)

        self.elu = nn.ELU()

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def decay_dropout(self):
        """Decay the dropout rate."""
        self.dropout_rate = max(0, self.dropout_rate * self.decay_rate)
        self.dropout.p = self.dropout_rate

    def forward(self, state):
        x = self.elu(self.fc1(state))
        if self.use_batch_norm or self.use_renorm:
            x = self.batch_norm1(x)
        x = self.elu(self.fc2(x))
        if self.use_batch_norm or self.use_renorm:
            x = self.batch_norm2(x)
        
        x = self.dropout(x)
        
        self.decay_dropout()

        return self.fc_out(x)



class CrossQ_VectorCritic(nn.Module):
    def __init__(self, state_dim, hidden_size=2048, num_heads=1, use_layer_norm=True, n_critics=2):
        super(CrossQ_VectorCritic, self).__init__()

        self.critics = nn.ModuleList([
            CrossQ_Critic(state_dim, hidden_size, num_heads, use_layer_norm) for _ in range(n_critics)
        ])

    def forward(self, s, a, fp):
        outputs = [critic(s, a, fp) for critic in self.critics]
        Q_list, A_list, G_list, reg_list = [], [], [], []
        
        for output in outputs:
            Q, A, G, reg = output
            Q_list.append(Q)
            A_list.append(A)
            G_list.append(G)
            reg_list.append(reg)

        # Stack the results from the critics
        Q_mean = torch.mean(torch.stack(Q_list, dim=0), dim=0)
        A_mean = torch.mean(torch.stack(A_list, dim=0), dim=0)
        G_mean = torch.mean(torch.stack(G_list, dim=0), dim=0)
        reg_mean = torch.mean(torch.stack(reg_list, dim=0), dim=0)
        return Q_mean, A_mean, G_mean, reg_mean


class CrossQ_Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=2048, num_heads=1, use_layer_norm=True, initial_dropout_rate=0.03, decay_rate=0.995):
        super(CrossQ_Critic, self).__init__()

        self.initial_dropout_rate = initial_dropout_rate
        self.dropout_rate = initial_dropout_rate  # current dropout rate (3% initially)
        self.decay_rate = decay_rate  # decay factor for dropout rate (0.995)

        # for ego critic
        self.ego_fc0 = nn.Linear(state_dim + 1, hidden_size)
        self.ego_fc1 = nn.Linear(hidden_size, hidden_size)
        self.ego_fc2 = nn.Linear(hidden_size, 1)

        # upstream critic
        self.u_fc1 = nn.Linear(hidden_size, hidden_size)
        self.u_fc2 = nn.Linear(hidden_size, hidden_size)
        self.u_fc3 = nn.Linear(hidden_size, 1)
        # downstream critic
        self.d_fc1 = nn.Linear(hidden_size, hidden_size)
        self.d_fc2 = nn.Linear(hidden_size, hidden_size)
        self.d_fc3 = nn.Linear(hidden_size, 1)
        # multihead attention layers
        self.u_attentions = [
            MultiHeadAttention(d_model=state_dim + 1 + 2, d_proj=hidden_size, num_heads=num_heads) for _ in range(1)
        ]
        for i, attention in enumerate(self.u_attentions):
            self.add_module('u_attention_{}'.format(i), attention)

        self.d_attentions = [
            MultiHeadAttention(d_model=state_dim + 1 + 2, d_proj=hidden_size, num_heads=num_heads) for _ in range(1)
        ]
        for i, attention in enumerate(self.d_attentions):
            self.add_module('d_attention_{}'.format(i), attention)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        # normalization for the critic
        if use_layer_norm:
            self.layer_norm_ego_1 = nn.LayerNorm(hidden_size)
            self.layer_norm_ego_2 = nn.LayerNorm(hidden_size)
            self.layer_norm_target_1 = nn.LayerNorm(hidden_size)
            self.layer_norm_target_2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def decay_dropout(self):
        """Decay the dropout rate."""
        self.dropout_rate = max(0, self.dropout_rate * self.decay_rate)
        self.dropout.p = self.dropout_rate

    def d_egat(self, x, adj):
        x = torch.cat([att(x, x, x, adj) for att in self.d_attentions], dim=1)
        x_target, x_others = x[0, :], torch.sum(x[1:, :], 0)
        return x_target, x_others

    def u_egat(self, x, adj):
        x = torch.cat([att(x, x, x, adj) for att in self.u_attentions], dim=1)
        x_target, x_others = x[0, :], torch.sum(x[1:, :], 0)
        return x_target, x_others

    def event_critic(self, fp):
        u_adjs, d_adjs, u_features, d_features = prepare_eg(fp)
        x_target, x_other, reg = [], [], []

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

            x_target.append(u_x_target + d_x_target)
            x_other.append(u_x_other + d_x_other)

        x_target = torch.stack(x_target, dim=0)
        x_other = torch.stack(x_other, dim=0)

        # upstream event critic network
        target = self.elu(self.layer_norm_target_1(self.u_fc1(x_target)))
        target = self.elu(self.layer_norm_target_2(self.u_fc2(target)))
        target = self.dropout(self.u_fc3(target))

        # downstream event critic network
        a = self.elu(self.layer_norm_target_1(self.d_fc1(x_other)))
        a = self.elu(self.layer_norm_target_2(self.d_fc2(a)))
        a = self.dropout(self.d_fc3(a))

        target = target.view(-1, 1)
        a = a.view(-1, 1)

        reg = torch.stack(reg, dim=0).view(-1, 1) if reg else torch.zeros(1)

        return target, a, reg

    def ego_critic(self, ego):
        out1 = self.relu(self.layer_norm_ego_1(self.ego_fc0(ego)))
        out1 = self.dropout(out1)
        out1 = self.relu(self.layer_norm_ego_2(self.ego_fc1(out1)))
        out1 = self.dropout(out1)  
        Q = self.ego_fc2(out1)
        return Q


    def forward(self, x, a, fp):
        ego = torch.cat([x, a], 1)
        A = self.ego_critic(ego)
        V_target, V_marginal, reg = self.event_critic(fp)
        G = A + V_target + V_marginal

        self.decay_dropout()

        return A, V_target + V_marginal, G.view(-1, 1), reg



class Agent(nn.Module):
    def __init__(self, state_dim, name='', n_stops=22, buslist=None, use_layer_norm=True, gamma=0.95, seed=123):
        super(Agent, self).__init__()
        random.seed(seed)
        self.name = name
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = 1
        self.n_stops = n_stops
        self.buslist = buslist
        self.seed = seed

        self.actor = CrossQ_Actor(input_size=state_dim, hidden_size=2048, output_size=self.action_dim, use_batch_norm=True, use_renorm=False)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0005, betas=(0.5, 0.999))

        self.critic = CrossQ_VectorCritic(state_dim=state_dim, hidden_size=2048, use_layer_norm=use_layer_norm, n_critics=2)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.0005, betas=(0.5, 0.999))

        self.noise_std = 0.2
        self.noise_decay = 0.99

    def choose_action(self, state, noise=True):
        state = torch.tensor(state, dtype=torch.float, device=next(self.actor.parameters()).device).unsqueeze(0)
        self.actor.eval()
        action = self.actor(state).squeeze(0).detach().cpu().numpy()

        if noise:
            action += self.noise_std * np.random.normal(size=action.shape)
            self.noise_std *= self.noise_decay

        self.actor.train()
        return action


    def lr_decay(self, decay_rate=0.99):
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] *= decay_rate
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] *= decay_rate

    def learn(self, memories, batch_size=16):
        if len(memories) < batch_size:
            return 0, 0

        batch = random.sample(memories, batch_size)
        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp = [], [], [], [], [], []

        for s, fp, a, r, ns, nfp in batch:
            batch_s.append(s)
            batch_fp.append(fp)
            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)
            batch_nfp.append(nfp)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        b_fp = [torch.tensor(f, dtype=torch.float) for f in batch_fp]
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_ns = torch.tensor(batch_ns, dtype=torch.float)
        b_nfp = [torch.tensor(nf, dtype=torch.float) for nf in batch_nfp]

        def critic_learn():
            Q, A, G, reg = self.critic(b_s, b_a, b_fp)
            Q_, A_, G_, _ = self.critic(b_ns, self.actor(b_ns).detach(), b_nfp)

            q_target = b_r + self.gamma * (G_.detach()).view(-1, 1)

            loss_fn = nn.MSELoss()
            qloss = loss_fn(G, q_target) + 0.1 * reg.mean()

            # Backpropagate and update critic
            self.critic_optim.zero_grad()
            qloss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()

            return qloss.item()

        def actor_learn():
            policy_loss, _, _, _ = self.critic(b_s, self.actor(b_s), b_fp)
            entropy = -torch.mean(torch.distributions.Normal(policy_loss, 1.0).entropy())  # Approximate entropy term
            policy_loss = -torch.mean(policy_loss) + 0.001 * entropy
            self.actor_optim.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optim.step()
            return policy_loss.item()

        critic_loss = critic_learn()
        actor_loss = actor_learn()

        self.lr_decay()

        return actor_loss, critic_loss

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))
        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

    def load(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))
        actor_path = f"{abspath}/save/{self.name}_{model}_{self.seed}_actor.pth"
        critic_path = f"{abspath}/save/{self.name}_{model}_{self.seed}_critic.pth"

        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def to(self, device):
        super(Agent, self).to(device)
        self.actor.to(device)
        self.critic.to(device)
        return self
