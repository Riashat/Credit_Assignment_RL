import numpy as np
import torch
import gym
import argparse
import os

import utils
import DDPG_Discrete as DDPG
from env_utils import AtariEnvWrapper



# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:

			#action = policy.select_action(np.array(obs))
			action = policy.select_action(np.array(obs)).clip(env.action_space.low, env.action_space.high)
			action  = np.random.choice(np.arange(action.shape[0]), p=action.ravel())
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward


def logprobs_and_entropy(self, x, actions):
	x = self(x)
	log_probs = F.log_softmax(x, dim=1)
	probs = F.softmax(x, dim=1)
	action_log_probs = log_probs.gather(1, actions)
	dist_entropy = -(log_probs * probs).sum(-1).mean()
	
	return action_log_probs, dist_entropy


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="DDPG")					# Policy name
	parser.add_argument("--env_name", default="PongNoFrameskip-v4")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--lambda_actor", default=0.0, type=float)			# Frequency of delayed policy updates
	args = parser.parse_args()

	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print ("---------------------------------------")
	print ("Settings: %s" % (file_name))
	print ("---------------------------------------")

	if not os.path.exists("./results2"):
		os.makedirs("./results2")
	if args.save_models and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")


	# env = gym.make(args.env_name)
	env = AtariEnvWrapper(args.env_name)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	max_action = 1

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	# max_action = int(env.action_space.high[0])

	import pdb; pdb.set_trace()

	# Initialize policy
	if args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)

	replay_buffer = utils.ReplayBuffer()
	
	# Evaluate untrained policy
	evaluations = [evaluate_policy(policy)] 

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	done = True 

	action_probs = np.array([])
	while total_timesteps < args.max_timesteps:
		
		if done: 

			if total_timesteps != 0: 
				print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
				policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.lambda_actor)
				action = policy.select_action(np.array(obs))
				action_choice  = np.random.choice(np.arange(action.shape[0]), p=action.ravel())	

				### for monitoring softmax saturation
				print ("Action Probabilities", action)
				print ("Chosen Action", action_choice)
			
			# Evaluate episode
			if timesteps_since_eval >= args.eval_freq:
				timesteps_since_eval %= args.eval_freq
				evaluations.append(evaluate_policy(policy))
				
				if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
				np.save("./results2/%s" % (file_name), evaluations) 
			
			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		"""
		We can do epsilon greedy here for discrete actions!
		"""
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = policy.select_action(np.array(obs))
			action_choice = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			action_choice  = np.random.choice(np.arange(action.shape[0]), p=action.ravel())	

		"""
		Ignorining OU Noise here - need to add this back - encourages exploration, or do epislon greedy!
		"""
		# if args.expl_noise != 0: 
		# 	action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

		# Perform action
		new_obs, reward, done, _ = env.step(action_choice) 
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward

		# Store data in replay buffer
		replay_buffer.add((obs, new_obs, action, reward, done_bool))

		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1
		
	# Final evaluation 
	evaluations.append(evaluate_policy(policy))
	if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	np.save("./results2/%s" % (file_name), evaluations)  


	
