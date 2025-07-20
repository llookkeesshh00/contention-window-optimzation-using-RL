import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.our_ddpg.PER import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x):
    """Converts x to a torch.FloatTensor on the correct device."""
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    
    # Ensure x is a NumPy array of float32
    if isinstance(x, list):
        x = np.array(x, dtype=np.float32)
    
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:  # Handle arrays of objects (e.g., lists of lists)
            x = np.array(x.tolist(), dtype=np.float32)
        
        return torch.tensor(x, dtype=torch.float, device=device)

    raise TypeError(f"Unsupported data type {type(x)} for conversion to tensor.")

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
        # Ensure state and action have the same number of dimensions
        if state.dim() == 2 and action.dim() == 3:
            action = action.squeeze(dim=-1)  # Adjust dimensions if necessary
        
        if state.dim() == 3 and action.dim() == 2:
            state = state.squeeze(dim=-1)

        sa = torch.cat([state, action], dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        return self.l3(q)

class AACHER(object):
    def __init__(self, state_dim, action_dim, max_action, discount, tau, hidden_sizes, 
                 num_actors=10, num_critics=10, actor_lr=3e-4, critic_lr=3e-4, rc_weight=0.005):
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
        self.rc_weight = rc_weight
        self.actor_loss = 0
        self.critic_loss = 0

    def select_action(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        state = to_tensor(state)
        actions = torch.stack([actor(state) for actor in self.actors])
        return actions.mean(dim=0).cpu().data.numpy().flatten()

    def train(self, replay_buffer, prioritized=True, beta_value=0.4, epsilon=1e-6, T=1, 
              batch_size=256, gamma=0.99, tau=0.005):
        # Initialize previous TD errors for smoothing
        previous_td_errors = None  

        for _ in range(T):
            # Sample replay buffer
            if prioritized: 
                experience = replay_buffer.sample(batch_size, beta_value)
                if len(experience) == 5:
                    s, a, r, s_new, done = experience
                    weights, batch_idxes = np.ones(len(r)), None
                elif len(experience) == 7:
                    s, a, r, s_new, done, weights, batch_idxes = experience
                else:
                    raise ValueError(f"Unexpected number of values returned by sample(): {len(experience)}")
                
                r = r.reshape(-1, 1)
                done = done.reshape(-1, 1)
                not_done = 1 - done
                weights = np.sqrt(weights)
            else:
                s, a, r, s_new, done = replay_buffer.sample(batch_size)
                r = r.reshape(-1, 1)
                done = done.reshape(-1, 1)
                not_done = 1 - done
                weights = np.ones_like(r)

            # Convert data to tensors
            state = to_tensor(s)
            action = to_tensor(a)
            next_state = to_tensor(s_new)
            reward = to_tensor(r)
            not_done = to_tensor(not_done)
            weights = to_tensor(weights)

            # Compute target Q-values
            with torch.no_grad():
                next_actions = torch.stack([actor(next_state) for actor in self.actors_target]).mean(dim=0)
                next_q_values = torch.stack([critic(next_state, next_actions) for critic in self.critics_target])
                target_Q = (reward + not_done * self.discount * next_q_values.mean(dim=0)).view(-1, 1)

            # Train critics
            self.critic_loss = 0
            for critic, optim in zip(self.critics, self.critics_optim):
                current_Q = critic(state, action)
                reg_term = self.rc_weight * torch.norm(list(critic.parameters())[0])
                loss = F.mse_loss(current_Q, target_Q) + reg_term
                self.critic_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
            self.critic_loss /= self.num_critics

            # Train actors
            self.actor_loss = 0
            for actor, optim in zip(self.actors, self.actors_optim):
                loss = -torch.stack([critic(state, actor(state)) for critic in self.critics]).mean()
                self.actor_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
            self.actor_loss /= self.num_actors

            # Soft update of target networks
            for i in range(self.num_actors):
                for param, target_param in zip(self.actors[i].parameters(), self.actors_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for i in range(self.num_critics):
                for param, target_param in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

