import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        return self.l3(q)

class AACHER(object):
    def __init__(self, state_dim, action_dim, max_action, discount, tau, hidden_sizes, num_actors, num_critics, actor_lr, critic_lr):
        self.actors = [Actor(state_dim, action_dim, max_action, hidden_sizes).to(device) for _ in range(num_actors)]
        self.actors_target = [copy.deepcopy(actor) for actor in self.actors]
        self.actors_optim = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        
        self.critics = [Critic(state_dim, action_dim, hidden_sizes).to(device) for _ in range(num_critics)]
        self.critics_target = [copy.deepcopy(critic) for critic in self.critics]
        self.critics_optim = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.num_actors = num_actors
        self.num_critics = num_critics
        self.actor_loss = 0
        self.critic_loss = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actions = torch.stack([actor(state) for actor in self.actors])
        return actions.mean(dim=0).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            next_actions = torch.stack([actor(next_state) for actor in self.actors_target])
            next_actions = next_actions.mean(dim=0)
            next_q_values = torch.stack([critic(next_state, next_actions) for critic in self.critics_target])
            target_Q = reward + not_done * self.discount * next_q_values.mean(dim=0)
        
        for critic, optim in zip(self.critics, self.critics_optim):
            current_Q = critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
            optim.zero_grad()
            critic_loss.backward()
            optim.step()
        
        for actor, optim in zip(self.actors, self.actors_optim):
            actor_loss = -torch.stack([critic(state, actor(state)) for critic in self.critics]).mean()
            optim.zero_grad()
            actor_loss.backward()
            optim.step()
        
        for i in range(self.num_actors):
            for param, target_param in zip(self.actors[i].parameters(), self.actors_target[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for i in range(self.num_critics):
            for param, target_param in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
