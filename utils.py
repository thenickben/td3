import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(rewards, file_name, smoothing_window = 10):
	scores = pd.DataFrame({'Scores':rewards})
	fig = plt.figure(figsize=(10,5))
	plt.grid(True)
	plt.style.use('seaborn-bright')
	rewards_smoothed = scores.rolling(smoothing_window, min_periods=smoothing_window).mean()
	plt.plot(rewards_smoothed)
	plt.xlabel("Episode")
	plt.ylabel("Episode Reward (Smoothed)")
	plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
	#plt.show()
	plt.savefig(file_name)

def save_kwargs(kwargs, file_name):
    with open(file_name, 'w') as f:
        for key in kwargs.keys():
            f.write("%s,%s\n"%(key,kwargs[key]))
        f.close()
    print('Test kwargs saved in', file_name)

def save_scores(scores, file_name):
    with open(file_name, 'w') as f:
        for score in scores:
            f.write("%s\n"%(score))
        f.close()
    print('Test scores saved in', file_name)