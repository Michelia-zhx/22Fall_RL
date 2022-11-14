import numpy as np
from abc import abstractmethod
import os
from email.policy import Policy

import torch
import torch.nn as nn
from torch.optim import Adam

class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass

# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model
		self.model = None

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		self.model.train(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict(data_batch)
		return label_predict


class MyDaggerAgent(DaggerAgent):
	def __init__(self, observation_space, action_dim, args):
		super(DaggerAgent, self).__init__()
		self.args = args
		self.model = PolicyModel(observation_space, args.feature_dim, action_dim)
		self.loss_fn = nn.CrossEntropyLoss()
		
		self.action_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:11, 7:12}
		self.action_map_inv = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 11:6, 12:7}
		self.model_path = os.path.join(os.getcwd(), 'models', args.model_dir)
		self.best_running_loss = np.inf
	
	def _process_data(self, ob):
		if len(ob.shape) == 3:
			ob = ob[None, ...]
		input_ob = torch.as_tensor(ob.transpose(0, 3, 1, 2)).float()
		# print("input_ob: ", input_ob.shape)
		return input_ob
	
	def _process_label(self, label_batch):
		label_batch = np.array(label_batch)
		label_batch = np.vectorize(self.action_map_inv.get)(label_batch)
		return label_batch
	
	def forward(self, ob):
		input = self._process_data(ob)
		# print("input: ", input.shape)
		action_logits = self.model(input)
		return action_logits
	
	def select_action(self, ob):
		action_logits = self.forward(ob)
		select_action = int(torch.argmax(action_logits, -1)[0].numpy())
		return self.action_map[select_action]
	
	def update(self, data_batch, label_batch):
		data_batch = self._process_data(np.array(data_batch))
		label_batch = self._process_label(label_batch)
		index = np.array([i for i in range(data_batch.shape[0])])

		num_update = data_batch.shape[0] // self.args.batch_size
		optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		lr_scheduler = self.create_lr_scheduler(optimizer, self.args)
		
		for epoch in range(self.args.num_epoch):
			running_loss = 0
			np.random.shuffle(index)
			epoch_data = data_batch[index]
			epoch_label = label_batch[index]
			
			for i in range(num_update):
				start = i * self.args.batch_size
				end = (i + 1) * self.args.batch_size
				minibatch_data = epoch_data[start:end]
				minibatch_label = torch.as_tensor(epoch_label[start:end]).long()
				minibatch_label = minibatch_data.type(torch.LongTensor)
				action_logits = self.model(minibatch_data)
				loss = self.loss_fn(action_logits, minibatch_label)
				running_loss += loss.item()
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				print("[Classifier Training] epoch: {}/{}, running loss: {}".format(epoch, self.args.num_epoch, running_loss))

				if running_loss <= self.best_running_loss:
					self.best_running_loss = running_loss
					self.save_model(self.model_path)
				else:
					break
			
			lr_scheduler.step()
	
	def beta_decay(self, i):
		return self.args.beta * (1 - i / (self.args.num_frames // self.args.num_steps))
	
	def create_lr_scheduler(self, optimizer, args):
		if args.lr_rampdown_epochs:
			assert args.lr_rampdown_epochs >= args.num_epoch
			lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr_rampdown_epochs)
		return lr_scheduler

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)


class PolicyModel(nn.Module):
	def __init__(self, input_shape, feature_dim, action_dim):
		super(PolicyModel, self).__init__()
		input_channel_dim = input_shape[-1]
		self.cnn = nn.Sequential(
			nn.Conv2d(input_channel_dim, 16, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
		)

		with torch.no_grad():
			sample_data = torch.as_tensor(
				np.random.randn(2, input_shape[2], input_shape[0], input_shape[1]),
			).float()
			flatten_dim = self.cnn(sample_data).shape[-1]
		
		self.action_layer = nn.Sequential(
			nn.Linear(flatten_dim, feature_dim),
			nn.ReLU(),
			nn.Linear(feature_dim, action_dim)
		)

	def forward(self, inputs):
		return self.action_layer(self.cnn(inputs))