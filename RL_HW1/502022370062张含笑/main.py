from arguments import get_args
from Dagger import DaggerAgent, ExampleAgent, MyDaggerAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os


def plot(record):
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(record['steps'], record['mean'],
	        color='blue', label='reward')
	ax.fill_between(record['steps'], record['min'], record['max'],
	                color='blue', alpha=0.2)
	ax.set_xlabel('number of steps')
	ax.set_ylabel('Average score per episode')
	ax1 = ax.twinx()
	ax1.plot(record['steps'], record['query'],
	         color='red', label='query')
	ax1.set_ylabel('queries')
	reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
	query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
	patch_set = [reward_patch, query_patch]
	ax.legend(handles=patch_set)
	fig.savefig('performance.png')


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
	def __init__(self, env_name, num_stacks):
		self.env = gym.make(env_name)
		# num_stacks: the agent acts every num_stacks frames
		# it could be any positive integer
		self.num_stacks = num_stacks
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def step(self, action):
		reward_sum = 0
		for stack in range(self.num_stacks):
			obs_next, reward, done, info = self.env.step(action)
			reward_sum += reward
			if done:
				self.env.reset()
				return obs_next, reward_sum, done, info
		return obs_next, reward_sum, done, info

	def reset(self):
		return self.env.reset()


def main():
	# load hyper parameters
	args = get_args()
	num_updates = int(args.num_frames // args.num_steps)
	start = time.time()
	record = {'steps': [0],
	          'max': [0],
	          'mean': [0],
	          'min': [0],
	          'query': [0]}
	# query_cnt counts queries to the expert
	query_cnt = 0

	# environment initial
	envs = Env(args.env_name, args.num_stacks)
	# action_shape is the size of the discrete action set, here is 18
	# Most of the 18 actions are useless, find important actions
	# in the tips of the homework introduction document
	raw_action_shape = envs.action_space.n
	action_shape = 8
	# observation_shape is the shape of the observation
	# here is (210,160,3)=(height, width, channels)
	observation_shape = envs.observation_space.shape
	# print(action_shape, observation_shape)

	# agent initial
	# you should finish your agent with DaggerAgent
	# e.g. agent = MyDaggerAgent()
	# agent = ExampleAgent()
	agent = MyDaggerAgent(observation_shape, action_shape, args)
	
	# You can play this game yourself for fun
	if args.play_game:
		obs = envs.reset()
		while True:
			im = Image.fromarray(obs)
			im.save('imgs/' + str('screen') + '.jpeg')
			action = int(input('input action'))
			while action < 0 or action >= raw_action_shape:
				action = int(input('re-input action'))
			obs_next, reward, done, _ = envs.step(action)
			obs = obs_next
			if done:
				obs = envs.reset()
	
	def get_expert_action(obs):
		"""show img to human and receive the action from human"""
		im = Image.fromarray(obs)
		im.save('imgs/' + str('screen') + '.jpeg')
		# envs.render()
		info = ""
		while True:
			try:
				input_action = input('input action' + info + ':')
				# print(input_action)
				if input_action == "exit":
					return -1
				elif input_action == "bad":
					return -2
				action = int(input_action)
				if action < 0 or action >= raw_action_shape:
					info = "re-input action :"
					continue
				break
			except:
				info = " (invalid input just now)"
		return action

	data_set = {'data': [], 'label': []}
	# start train your agent
	for i in range(num_updates):
		# an example of interacting with the environment
		# we init the environment and receive the initial observation
		obs = envs.reset()
		# print("envs reset")
		expert_action, labels, valid_steps = None, [], []
		beta = agent.beta_decay(i)
		# print(beta)
		# we get a trajectory with the length of args.num_steps
		for step in range(args.num_steps):
			bad_frame = False
			# Sample actions
			if np.random.rand() < beta:
				if expert_action == None:
					action = get_expert_action(obs)
					if action == -2:
						# expert thought it's a bad frame, whatever action is ok
						action = agent.select_action(obs)
				else:
					action = expert_action
				query_cnt += 1
				# # we choose a random action
				# action = envs.action_space.sample()
			else:
				# we choose a special action according to our model
				action = agent.select_action(obs)

			# interact with the environment
			# we input the action to the environments and it returns some information
			# obs_next: the next observation after we do the action
			# reward: (float) the reward achieved by the action
			# down: (boolean)  whether it’s time to reset the environment again.
			#           done being True indicates the episode has terminated.
			obs_next, reward, done, _ = envs.step(action)
			# we view the new observation as current observation
			obs = obs_next
			# if the episode has terminated, we need to reset the environment.
			if done:
				envs.reset()

			# an example of saving observations
			if args.save_img:
				im = Image.fromarray(obs)
				im.save('imgs/' + str(step) + '.jpeg')
			data_set['data'].append(obs)

			# an example of saving expert actions
			expert_action = get_expert_action(obs)
			if expert_action == -2:
				bad_frame = True
				data_set['data'].pop()
				expert_action = agent.select_action(obs)
				break
			
			if not bad_frame:
				labels.append(expert_action)
				valid_steps.append(step)

		# You need to label the images in 'imgs/' by recording the right actions in label.txt
		with open('imgs/label.txt', 'w') as f:
			f.write('\n'.join([f"{valid_steps[i]} {labels[i]}" for i in range(len(labels))]))

		# After you have labeled all the images, you can load the labels
		# for training a model
		with open('imgs/label.txt', 'r') as f:
			for label_tmp in f.readlines():
				label_tmp = int(label_tmp.strip().split()[1])
				data_set['label'].append(label_tmp)

		# design how to train your model with labeled data
		agent.update(data_set['data'], data_set['label'])

		if (i + 1) % args.log_interval == 0:
			total_num_steps = (i + 1) * args.num_steps
			obs = envs.reset()
			reward_episode_set = []
			reward_episode = 0
			# evaluate your model by testing in the environment
			for step in range(args.test_steps):
				action = agent.select_action(obs)
				# you can render to get visual results
				# envs.render()
				obs_next, reward, done, _ = envs.step(action)
				reward_episode += reward
				obs = obs_next
				if done:
					reward_episode_set.append(reward_episode)
					reward_episode = 0
					envs.reset()

			end = time.time()
			print(
				"TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
					.format(
					time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
					i, total_num_steps,
					int(total_num_steps / (end - start)),
					query_cnt,
					np.mean(reward_episode_set),
					np.min(reward_episode_set),
					np.max(reward_episode_set)
				))
			record['steps'].append(total_num_steps)
			record['mean'].append(np.mean(reward_episode_set))
			record['max'].append(np.max(reward_episode_set))
			record['min'].append(np.min(reward_episode_set))
			record['query'].append(query_cnt)
			plot(record)

			path = os.path.join(os.getcwd(), 'models', str(i)+'_model.pth')
			agent.save_model(path)


if __name__ == "__main__":
	main()