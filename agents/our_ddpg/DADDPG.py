import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes = [128, 128]):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], action_dim)
		"""
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		"""

		self.max_action = max_action


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes = [128, 128]):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)
		"""
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)
		"""


	def forward(self, state, action):
		if(len(state.shape) == 3):
			sa = torch.cat([state, action], 2)
		else:
			sa = torch.cat([state, action], 1)
		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.9, tau=1e-1):  # discount = 0.99, tau 0.005
		self.actor1 = Actor(state_dim, action_dim, max_action).to(device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=3e-4)  # 3e-4

		self.actor2 = Actor(state_dim, action_dim, max_action).to(device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=4e-3)  # 3e-4
		
		self.max_action = max_action
		self.actor_loss = 0
		self.critic_loss = 0
		self.policy_noise = 0.2
		self.noise_clip = 0.5
		self.total_it = 0
		self.discount = discount
		self.tau = tau


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		action1 = self.actor1(state)
		action2 = self.actor2(state)

		q1 = self.critic(state, action1)
		q2 = self.critic(state, action2)

		action = action1 if q1 >= q2 else action2

		return action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=64):  # batch_size = 256
		# Sample replay buffer
		self.total_it += 1
		update_a1 = True if self.total_it % 2 == 0 else False
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			next_action1 = self.actor1_target(next_state)
			next_action2 = self.actor2_target(next_state)

			noise = torch.randn(
				(action.shape[0], action.shape[1]),
				dtype=action.dtype, layout=action.layout, device=device
			) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_action1 = (next_action1 + noise).clamp(-self.max_action, self.max_action)
			next_action2 = (next_action2 + noise).clamp(-self.max_action, self.max_action)

			next_Q_a1 = self.critic_target(next_state, next_action1)
			next_Q_a2 = self.critic_target(next_state, next_action2)
			next_Q = torch.min(next_Q_a1, next_Q_a2)

			target_Q = reward + not_done * self.discount * next_Q

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)
		self.critic_loss = critic_loss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if(update_a1):
			actor1_loss = -self.critic(state, self.actor1(state)).mean()

			self.actor1_optimizer.zero_grad()
			actor1_loss.backward()
			self.actor1_optimizer.step()

			for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			actor2_loss = -self.critic(state, self.actor2(state)).mean()

			self.actor2_optimizer.zero_grad()
			actor2_loss.backward()
			self.actor2_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
			for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor1.state_dict(), filename + "_actor1")
		torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")

		torch.save(self.actor2.state_dict(), filename + "_actor2")
		torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor1.load_state_dict(torch.load(filename + "_actor1"))
		self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer1"))
		self.actor1_target = copy.deepcopy(self.actor1)

		self.actor2.load_state_dict(torch.load(filename + "_actor2"))
		self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer2"))
		self.actor2_target = copy.deepcopy(self.actor2)
