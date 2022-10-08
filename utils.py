import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomBuffer(object):
	def __init__(self, state_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0 #store index
		self.size = 0 #current size of buffer

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dw = np.zeros((max_size, 1))  #mask of dead&win

		self.device = device

	def add(self, state, action, reward, next_state, dw):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw #0,0,0，...，1

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		with torch.no_grad():
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.Tensor(self.action[ind]).long().to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.dw[ind]).to(self.device)
			)

def evaluate_policy(env, model, render, turns = 3):
    scores = 0
    for j in range(turns):
        s, done= env.reset(), False
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_next, r, done, info = env.step(a)
            scores += r
            s = s_next
            if render:
                env.render()
    return scores/turns


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')