import numpy as np
import torch
import gym
import argparse
import os
import time
import datetime

import utils
import TD3
import OurDDPG
import DDPG
from schedules import LinearSchedule

import replay_buffer as rb

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="TD3")							# Policy name
	parser.add_argument("--env_name", default="Pendulum-v0")					# OpenAI gym environment name
	parser.add_argument("--replay_buffer", default="prioritized")				# Replay Buffer type
	parser.add_argument("--replay_buffer_size", default=5e4, type=int)			# Replay Buffer capacity
	parser.add_argument("--replay_buffer_alpha", default=0.6, type=float)		# Replay Buffer prioritization weight
	parser.add_argument("--seed", default=0, type=int)							# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)				# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=1e3, type=float)					# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=5e4, type=float)				# Max time steps to run environment for
	parser.add_argument("--save_models", default="True", type=bool)				# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)				# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)					# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)					# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)						# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)				# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)				# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)					# Frequency of delayed policy updates
	parser.add_argument("--lr_actor", default=0.001, type=float)				# Learning rate of actor
	parser.add_argument("--lr_critic", default=0.001, type=float)				# Learning rate of critic
	parser.add_argument("--prioritized_replay_eps", default=1e-3, type=float)	# Replay Buffer epsilon (PRE)
	parser.add_argument("--prioritized_replay_beta0", default=0.4, type=float)	# Replay Buffer initial beta (PRE)
	args = parser.parse_args()

#Training kwargs
	kwargs = {  "policy_name": args.policy_name,
				"env_name": args.env_name,
				"replay_buffer": args.replay_buffer,
				"replay_buffer_size": args.replay_buffer_size,
				"replay_buffer_alpha": args.replay_buffer_alpha,
				"seed": args.seed,
				"start_timesteps": args.start_timesteps,
				"eval_freq": args.eval_freq,
				"max_timesteps": args.max_timesteps,
				"save_models": args.save_models,
				"expl_noise": args.expl_noise,
				"batch_size": args.batch_size,
				"discount": args.discount,
				"tau": args.tau,
				"policy_noise": args.policy_noise,
				"noise_clip": args.noise_clip,
				"policy_freq": args.policy_freq,
				"lr_actor": args.lr_actor,
				"prioritized_replay_eps": args.prioritized_replay_eps,
				"prioritized_replay_beta0": args.prioritized_replay_beta0
         }

	# cls
	os.system('cls' if os.name == 'nt' else 'clear')

	if not os.path.exists("./results"):
    		os.makedirs("./results")
	if args.save_models and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")

	# Time stamp for repeated test names
	ts = time.time()
	ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

	test_name = "%s_%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed), ts)
	plot_name = "%s_%s_%s_%s_plot.png" % (args.policy_name, args.env_name, str(args.seed), ts)
	kwargs_name = "%s_%s_%s_%s_kwargs.csv" % (args.policy_name, args.env_name, str(args.seed), ts)
	scores_name = "%s_%s_%s_%s_scores.csv" % (args.policy_name, args.env_name, str(args.seed), ts)

	print("---------------------------------------")
	print("Settings: %s" % (test_name))
	utils.save_kwargs(kwargs, "./results/%s" % (kwargs_name))
	print("---------------------------------------")

	# Environment and Agent instantiation

	env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Instantiate Replay Buffer	
	if args.replay_buffer == "vanilla": 
		replay_buffer = rb.ReplayBuffer(size = args.replay_buffer_size)
		PER = False
	elif args.replay_buffer == "prioritized": 
		replay_buffer = rb.PrioritizedReplayBuffer(size = int(np.round(np.sqrt(args.replay_buffer_size))), 
												   alpha = args.replay_buffer_alpha)
		PER = True
		prioritized_replay_beta_iters = args.max_timesteps
		prioritized_replay_beta0 = args.prioritized_replay_beta0
		beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p = prioritized_replay_beta0,
                                       final_p = 1.0)

	# Instantiate policy
	if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action, args.lr_actor, args.lr_critic, PER, args.prioritized_replay_eps)
	elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action, args.lr_actor, args.lr_critic, PER, args.prioritized_replay_eps)

	# Evaluate untrained policy
	evaluations = [evaluate_policy(env, policy)] 

	# Training loop #######################################

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	episode_rewards = []
	done = True 

	while total_timesteps < args.max_timesteps:
		
		if done: 

			if total_timesteps != 0: 
				print('Total T: {} Episode Num: {} Episode T: {} Reward: {}'.format(total_timesteps, episode_num, episode_timesteps, episode_reward))
				episode_rewards.append(episode_reward)
				
				# PER Beta scheduled update 
				if PER: beta = beta_schedule.value(total_timesteps)
				else: beta = 0.
				# Policy update step
				if args.policy_name == "TD3":
					policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq, beta)
				else: 
					policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, beta)
			
			# Evaluate episode
			if timesteps_since_eval >= args.eval_freq:
				timesteps_since_eval %= args.eval_freq
				evaluations.append(evaluate_policy(env, policy))
				
				# save evaluation
				#if args.save_models: policy.save(test_name, directory="./pytorch_models")
				#np.save("./results/%s" % (test_name), evaluations) 
			
			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			if args.expl_noise != 0: 
				action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

		# Perform action
		new_obs, reward, done, _ = env.step(action) 
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward

		# Push experience into replay buffer
		experience = (obs, action, reward, new_obs, done_bool)
		replay_buffer.add(experience)

		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1
		
	# Final evaluation 
	evaluations.append(evaluate_policy(env, policy))
	
	# Save results
	if args.save_models: policy.save("%s" % (test_name), directory="./pytorch_models")
	#np.save("./results/%s" % (evaluations_file), evaluations)  
	#np.save("./results/%s" % ('rewards.txt'), episode_rewards) 
	utils.save_scores(episode_rewards, "./results/%s" % (scores_name))
	utils.plot(episode_rewards, "./results/%s" % (plot_name), 1)

if __name__ == "__main__":
	main()