import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    else:
        return torch.tensor(x, dtype=torch.float, device=device)

class DynamicPERBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = []

    def add(self, experience, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([])), np.array([]), np.array([])

        batch_size = min(batch_size, len(self.buffer))
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()

        s, a, r, s_new, done = zip(*samples)

         # Ensure return type is (tuple, weights, indices)
        return (np.array(s), np.array(a), np.array(r), np.array(s_new), np.array(done)), np.array(weights), np.array(indices)

    
    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + self.epsilon) ** self.alpha

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

class AACHER:
    def __init__(self, state_dim, action_dim, max_action, discount, tau, hidden_sizes,
                 num_actors=10, num_critics=10, actor_lr=3e-4, critic_lr=3e-4, rc_weight=0.005,
                 buffer_size=100000):
        self.actors = [Actor(state_dim, action_dim, max_action, hidden_sizes).to(device) for _ in range(num_actors)]
        self.actors_target = [copy.deepcopy(actor) for actor in self.actors]
        self.actors_optim = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        
        self.critics = [Critic(state_dim, action_dim, hidden_sizes).to(device) for _ in range(num_critics)]
        self.critics_target = [copy.deepcopy(critic) for critic in self.critics]
        self.critics_optim = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.buffer = DynamicPERBuffer(buffer_size)
        self.discount = discount
        self.tau = tau
        self.num_actors = num_actors
        self.num_critics = num_critics
        self.rc_weight = rc_weight
        self.actor_loss = 0
        self.critic_loss = 0

    def select_action(self, state):
        state = to_tensor(state)
        actions = torch.stack([actor(state) for actor in self.actors])
        return actions.mean(dim=0).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        (s, a, r, s_new, done), weights, indices = replay_buffer.sample(batch_size)
        if len(s) == 0:
            return  
    
        state = to_tensor(s)
        action = to_tensor(a)
        reward = to_tensor(r).unsqueeze(1)
        next_state = to_tensor(s_new)
        not_done = to_tensor(1 - done).unsqueeze(1)
        weights = to_tensor(weights).unsqueeze(1)

        with torch.no_grad():
            next_actions = torch.stack([actor(next_state) for actor in self.actors_target]).mean(dim=0)
            next_q_values = torch.stack([critic(next_state, next_actions) for critic in self.critics_target])
            target_Q = reward + not_done * self.discount * next_q_values.mean(dim=0)

        self.critic_loss = 0
        td_errors = []
        for critic, optim in zip(self.critics, self.critics_optim):
            current_Q = critic(state, action)
            td_error = current_Q - target_Q
            td_errors.append(td_error.detach().cpu().numpy())
            loss = ((F.mse_loss(current_Q, target_Q) + self.rc_weight * torch.norm(list(critic.parameters())[0])) * weights).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.critic_loss += loss.item()

        self.actor_loss = 0
        for actor, optim in zip(self.actors, self.actors_optim):
            loss = -torch.stack([critic(state, actor(state)) for critic in self.critics]).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.actor_loss += loss.item()

        # Convert td_errors to NumPy array before updating priorities
        td_errors = np.abs(np.concatenate(td_errors)).mean(axis=0)
        replay_buffer.update_priorities(indices, td_errors)

