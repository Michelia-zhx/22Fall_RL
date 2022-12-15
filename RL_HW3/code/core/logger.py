import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.misc
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
# from scipy.interpolate import spline, interp1d

from core.util import time_seq

plt.rcParams.update({'font.size': 13})
plt.rcParams['figure.figsize'] = 10, 8


class TensorBoardLogger(object):
	"""docstring for Tensorboard_logger"""
	def __init__(self, log_dir):
		self.log_dir = log_dir
		self.writer = SummaryWriter(log_dir)

	def scalar_summary(self, tag, step, value):
		"""Log a scalar variable."""
		self.writer.add_scalar(tag, value, step)

	def text_summary(self, name, value):
		"""Log a text variable."""
		self.writer.add_text(name, value)

	def image_summary(self, tag, images, step):
		"""Log a list of images."""
		for i, img in enumerate(images):
			# Write the image to a string
			try:
				s = StringIO()
			except:
				s = BytesIO()
			scipy.misc.toimage(img).save(s, format="png")

			self.writer.add_image(tag, img, step)


class Plot:
	def __init__(self, save_path):
		self.Y = []
		self.X = []
		self.ax = None
		self.fig = None
		self.save_path = save_path

	def save(self):
		# list and ',' = list[0]
		line, = self.ax.plot(self.X, self.Y, 'b')
		if self.ax.get_title() != '':
			name = self.ax.get_title().replace(' ', '_')
			self.fig.savefig(self.save_path + name + '.png')
		else:
			name = time_seq()
			self.fig.savefig(self.save_path + name + '.png')


class MatplotlibLogger:

	def __init__(self, save_path):
		self.save_path = save_path
		self.plot_dict = {}

		if self.save_path[-1] != '/':
			self.save_path += '/'

	def add_plot(self, tag: str, x_label, y_label, title=''):
		plot = Plot(self.save_path)
		plot.fig, plot.ax = plt.subplots()
		plot.ax.set_xlabel(x_label)
		plot.ax.set_ylabel(y_label)
		plot.ax.set_title(title)

		self.plot_dict[tag] = plot

	def scalar_summary(self, tag, x, y):
		self.plot_dict[tag].Y.append(y)
		self.plot_dict[tag].X.append(x)
		self.plot_dict[tag].save()