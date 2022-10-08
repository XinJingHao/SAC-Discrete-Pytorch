import copy
import torch
import numpy as np
import torch.nn as nn
from utils import device
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def build_net(layer_shape, hid_activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hid_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]

		self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2


class Policy_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Policy_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.P = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		logits = self.P(s)
		probs = F.softmax(logits, dim=1)
		return probs


class SACD_Agent(object):
	def __init__(self, opt):
		self.action_dim = opt.action_dim
		self.batch_size = opt.batch_size
		self.gamma = opt.gamma
		self.tau = 0.005

		self.actor = Policy_Net(opt.state_dim, opt.action_dim, opt.hid_shape).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=opt.lr)

		self.q_critic = Q_Net(opt.state_dim, opt.action_dim, opt.hid_shape).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=opt.lr)

		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False

		self.alpha = opt.alpha
		self.adaptive_alpha = opt.adaptive_alpha
		if opt.adaptive_alpha:
			self.target_entropy = 0.3 * (-np.log(1 / opt.action_dim))  # H(discrete)>0
			self.log_alpha = torch.tensor(np.log(opt.alpha), dtype=float, requires_grad=True, device=device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=opt.lr)

		self.H_mean = 0

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor([state]).to(device) #from (s_dim,) to (1, s_dim)
			probs = self.actor(state)
			if deterministic:
				a = probs.argmax(-1).item()
			else:
				a = Categorical(probs).sample().item()
			return a

	def train(self,replay_buffer):
		s, a, r, s_next, dw = replay_buffer.sample(self.batch_size)

		#------------------------------------------ Train Critic ----------------------------------------#
		'''Compute the target soft Q value'''
		with torch.no_grad():
			next_probs = self.actor(s_next) #[b,a_dim]
			next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]
			next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
			min_next_q_all = torch.min(next_q1_all, next_q2_all)
			v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
			target_Q = r + (1 - dw) * self.gamma * v_next

		'''Update soft Q net'''
		q1_all, q2_all = self.q_critic(s) #[b,a_dim]
		q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) #[b,1]
		q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#------------------------------------------ Train Actor ----------------------------------------#
		for params in self.q_critic.parameters():
			#Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
			params.requires_grad = 	False

		probs = self.actor(s) #[b,a_dim]
		log_probs = torch.log(probs + 1e-8) #[b,a_dim]
		with torch.no_grad():
			q1_all, q2_all = self.q_critic(s)  #[b,a_dim]
		min_q_all = torch.min(q1_all, q2_all)

		a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=True) #[b,1]

		self.actor_optimizer.zero_grad()
		a_loss.mean().backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = 	True

		#------------------------------------------ Train Alpha ----------------------------------------#
		if self.adaptive_alpha:
			with torch.no_grad():
				self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
			alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			self.alpha = self.log_alpha.exp().item()

		#------------------------------------------ Update Target Net ----------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, index, b_envname):
		torch.save(self.actor.state_dict(), "./model/sacd_actor_{}_{}.pth".format(index, b_envname))
		torch.save(self.q_critic.state_dict(), "./model/sacd_critic_{}_{}.pth".format(index, b_envname))


	def load(self, index, b_envname):
		self.actor.load_state_dict(torch.load("./model/sacd_actor_{}_{}.pth".format(index, b_envname)))
		self.q_critic.load_state_dict(torch.load("./model/sacd_critic_{}_{}.pth".format(index, b_envname)))
