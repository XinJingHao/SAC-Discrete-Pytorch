import argparse
import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
from utils import evaluate_policy,str2bool, RandomBuffer
from SACD import SACD_Agent

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=50, help='which model to load')

parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e4, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
opt = parser.parse_args()
print(opt)

def main():
	#create Env
	EnvName = ['CartPole-v1','LunarLander-v2']
	BriefEnvName = ['CPV1', 'LLdV2']
	env = gym.make(EnvName[opt.EnvIdex])
	eval_env = gym.make(EnvName[opt.EnvIdex])
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = env._max_episode_steps

	#Seed everything
	torch.manual_seed(opt.seed)
	env.seed(opt.seed)
	env.action_space.seed(opt.seed)
	eval_env.seed(opt.seed)
	eval_env.action_space.seed(opt.seed)
	np.random.seed(opt.seed)

	print('Algorithm: SACD','  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
		  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

	if opt.write:
		timenow = str(datetime.now())[0:-10]
		timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
		writepath = 'runs/SACD_{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
		if os.path.exists(writepath): shutil.rmtree(writepath)
		writer = SummaryWriter(log_dir=writepath)

	#Build model and replay buffer
	if not os.path.exists('model'): os.mkdir('model')
	model = SACD_Agent(opt)
	if opt.Loadmodel: model.load(opt.ModelIdex, BriefEnvName[opt.EnvIdex])
	buffer = RandomBuffer(opt.state_dim, max_size=int(1e6))

	if opt.render:
		score = evaluate_policy(eval_env, model, True, 5)
		print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
	else:
		total_steps = 0
		while total_steps < opt.Max_train_steps:
			s, done, ep_r, steps = env.reset(), False, 0, 0
			while not done:
				steps += 1  # steps in current episode

				# interact with Env
				if buffer.size < opt.random_steps:
					a = env.action_space.sample()
				else:
					a = model.select_action(s, deterministic=False)
				s_next, r, done, info = env.step(a)

				'''Avoid impacts caused by reaching max episode steps'''
				if (done and steps != opt.max_e_steps):
					dw = True  # dw: dead and win
				else:
					dw = False

				# good for LunarLander
				if opt.EnvIdex == 1:
					if r <= -100:
						r = -10
						dw = True

				buffer.add(s, a, r, s_next, dw)
				s = s_next
				ep_r += r

				'''update if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
					for j in range(opt.update_every):
						model.train(buffer)

				'''record & log'''
				if (total_steps) % opt.eval_interval == 0:
					score = evaluate_policy(eval_env, model, render=False)
					if opt.write:
						writer.add_scalar('ep_r', score, global_step=total_steps)
						writer.add_scalar('alpha', model.alpha, global_step=total_steps)
						writer.add_scalar('H_mean', model.H_mean, global_step=total_steps)
					print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed,
						  'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
				total_steps += 1

				'''save model'''
				if (total_steps) % opt.save_interval == 0:
					model.save(int(total_steps/1000), BriefEnvName[opt.EnvIdex])
	env.close()
	eval_env.close()


if __name__ == '__main__':
	main()

