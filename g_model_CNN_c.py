import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

class Residual_block(nn.Module):
	def __init__(self, nb_filts):
		super(Residual_block, self).__init__()

		self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
		self.bn1 = nn.BatchNorm1d(num_features = nb_filts[1])
		#self.lrelu = nn.LeakyReLU()
		self.lrelu = nn.LeakyReLU(negative_slope = 0.3)
		self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[2],
			padding = 1,
			kernel_size = 3,
			stride = 1)
		self.bn2 = nn.BatchNorm1d(num_features = nb_filts[2])
		self.mp = nn.MaxPool1d(3)

		if nb_filts[0] != nb_filts[2]:
			self.downsample = True
			self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[2],
				padding = 1,
				kernel_size = 3,
				stride = 1)
		else:
			self.downsample = False


	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.lrelu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample:
			identity = self.conv_downsample(identity)
		
		out += identity
		out = self.lrelu(out)
		out = self.mp(out)
		return out



		

class raw_CNN_c(nn.Module):
	#def __init__(self, nb_filts, kernels, strides, name, first = False, downsample = False):
	def __init__(self, d_args):
		super(raw_CNN_c, self).__init__()
		self.first_conv = nn.Conv1d(in_channels = d_args['in_channels'],
			out_channels = d_args['filts'][0],
			kernel_size = d_args['first_conv'],
			padding = 0,
			stride = d_args['first_conv'])
		self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
		self.first_lrelu = nn.LeakyReLU(negative_slope = 0.3)

		self.block0 = Residual_block(nb_filts = d_args['filts'][1])
		d_args['filts'][1][0] = d_args['filts'][1][1]
		self.block1 = Residual_block(nb_filts = d_args['filts'][1])

		self.block2 = Residual_block(nb_filts = d_args['filts'][2])
		d_args['filts'][2][0] = d_args['filts'][2][1]
		self.block3 = Residual_block(nb_filts = d_args['filts'][2])
		self.block4 = Residual_block(nb_filts = d_args['filts'][2])
		self.block5 = Residual_block(nb_filts = d_args['filts'][2])

		self.block6 = Residual_block(nb_filts = d_args['filts'][3])

		self.last_conv = nn.Conv1d(in_channels = d_args['filts'][3][-1],
			out_channels = d_args['filts'][4],
			kernel_size = 1,
			padding = 3,
			stride = 1)
		self.last_bn = nn.BatchNorm1d(num_features = d_args['filts'][4])
		self.last_lrelu = nn.LeakyReLU(negative_slope = 0.3)

		self.global_maxpool = nn.AdaptiveMaxPool1d((1))
		self.global_avgpool = nn.AdaptiveAvgPool1d((1))

		self.fc1 = nn.Linear(in_features = d_args['filts'][-1] * 2,
			out_features = d_args['nb_fc_node'])
		self.fc2 = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],
			bias = False)


	def forward(self, x):
		x = self.first_conv(x)
		x = self.first_bn(x)
		x = self.first_lrelu(x)

		x = self.block0(x)
		x = self.block1(x)

		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)

		x = self.block6(x)

		x = self.last_conv(x)
		x = self.last_bn(x)
		x = self.last_lrelu(x)
		
		x_avg = self.global_avgpool(x)
		channel_dim = x_avg.size(1)
		x_avg = x_avg.view(-1, channel_dim)
		x_max = self.global_maxpool(x)
		x_max = x_max.view(-1, channel_dim)
		x = torch.cat((x_avg, x_max), dim = 1)

		x = self.fc1(x)
		y = self.fc2(x)

		return x, y




