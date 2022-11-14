import numpy as np

from abc import abstractmethod

class QAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass

class myQAgent(QAgent):
	def __init__(self, action_shape, state_shape, args):
		self.alpha = args.lr
		self.gamma = args.gamma
		self.Q = np.zeros((*state_shape, action_shape))

	def select_action(self, ob):
		ob = ob.astype(np.int32)
		q_values = self.Q[ob[0], ob[1]]
		action = np.argmax(q_values)
		return action

	def update(self, ob, action, reward, ob_next, done):
		ob, ob_next = ob.astype(np.int32), ob_next.astype(np.int32)
		target = reward + self.gamma * np.max(self.Q[ob_next[0], ob_next[1]])
		self.Q[ob[0], ob[1], action] += self.alpha * (target - self.Q[ob[0], ob[1], action])
