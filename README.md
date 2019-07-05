# DDPG and TD3 with Prioritized Experience Replay

PyTorch implementations of both Twin Delayed Deep Deterministic Policy Gradients (TD3) and Deep Deterministic Policy Gradients (DDPG), based on [codes](https://github.com/sfujim/TD3/) for DDPG and TD3 (plus some tweaks to make them work on Python3), OpenAI's [code](https://arxiv.org/abs/1511.05952) for an efficient implementation of Prioritized Experience Replay (PER) (with some tweaks), and using some [common baselines](https://github.com/openai/baselines/blob/master/baselines/common/) from OpenAI gym.


Method is tested on continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.0.1](https://github.com/pytorch/pytorch) and Python 3.6.8. 

### Usage

To run experiments on single environments can be run by calling:
```
python main.py --env Pendulum-v0
```
All hyperparameters can be accessed through parsing options:
```
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
```

### References

[DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
[TD3 paper](https://arxiv.org/abs/1802.09477)
[PER paper](https://arxiv.org/abs/1511.05952)

### Future work

Working on adding more exploration options (Parameter Space Noise), parallel environments and multi-agent environment support
