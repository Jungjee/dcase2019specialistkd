import os
import yaml
import struct
import pickle as pk
import numpy as np
import torch
import torch.nn as nn

from comet_ml import Experiment
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from torch.utils import data
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from copy import deepcopy

from g_model_CNN_c import raw_CNN_c

def get_utt_list(src_dir):
	'''
	Designed for DCASE2019 task 1-a
	'''
	l_utt = []
	for r, ds, fs in os.walk(src_dir):
		for f in fs:
			if f[-3:] != 'npy':
				continue
			k = f.split('.')[0]
			l_utt.append(k)

	return l_utt

def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CenterLoss(nn.Module):
	"""Center loss.
	
	Reference:
	Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
	
	Args:
		num_classes (int): number of classes.
		feat_dim (int): feature dimension.
	"""
	def __init__(self, num_classes = None, feat_dim = None, use_gpu = True, device = None):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu
		self.device = device

		if self.use_gpu:
			#self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long()
		#if self.use_gpu: classes = classes.cuda()
		if self.use_gpu: classes = classes.to(self.device)
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = distmat * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

		return loss

def summary(model, input_size, batch_size=-1, device="cuda", print_fn = None):
	if print_fn == None: printfn = print

	def register_hook(module):

		def hook(module, input, output):
			class_name = str(module.__class__).split(".")[-1].split("'")[0]
			module_idx = len(summary)

			m_key = "%s-%i" % (class_name, module_idx + 1)
			summary[m_key] = OrderedDict()
			summary[m_key]["input_shape"] = list(input[0].size())
			summary[m_key]["input_shape"][0] = batch_size
			if isinstance(output, (list, tuple)):
				summary[m_key]["output_shape"] = [
					[-1] + list(o.size())[1:] for o in output
				]
			else:
				summary[m_key]["output_shape"] = list(output.size())
				summary[m_key]["output_shape"][0] = batch_size

			params = 0
			if hasattr(module, "weight") and hasattr(module.weight, "size"):
				params += torch.prod(torch.LongTensor(list(module.weight.size())))
				summary[m_key]["trainable"] = module.weight.requires_grad
			if hasattr(module, "bias") and hasattr(module.bias, "size"):
				params += torch.prod(torch.LongTensor(list(module.bias.size())))
			summary[m_key]["nb_params"] = params

		if (
			not isinstance(module, nn.Sequential)
			and not isinstance(module, nn.ModuleList)
			and not (module == model)
		):
			hooks.append(module.register_forward_hook(hook))

	device = device.lower()
	#'''
	assert device in [
		"cuda",
		"cpu",
	], "Input device is not valid, please specify 'cuda' or 'cpu'"
	#'''

	#dtype = torch.cuda.FloatTensor
	#'''
	if device == "cuda" and torch.cuda.is_available():
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor
	#'''

	# multiple inputs to the network
	if isinstance(input_size, tuple):
		input_size = [input_size]

	# batch_size of 2 for batchnorm
	x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
	# print(type(x[0]))

	# create properties
	summary = OrderedDict()
	hooks = []

	# register hook
	model.apply(register_hook)

	# make a forward pass
	# print(x.shape)
	model(*x)

	# remove these hooks
	for h in hooks:
		h.remove()

	print_fn("----------------------------------------------------------------")
	line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
	print_fn(line_new)
	print_fn("================================================================")
	total_params = 0
	total_output = 0
	trainable_params = 0
	for layer in summary:
		# input_shape, output_shape, trainable, nb_params
		line_new = "{:>20}  {:>25} {:>15}".format(
			layer,
			str(summary[layer]["output_shape"]),
			"{0:,}".format(summary[layer]["nb_params"]),
		)
		total_params += summary[layer]["nb_params"]
		total_output += np.prod(summary[layer]["output_shape"])
		if "trainable" in summary[layer]:
			if summary[layer]["trainable"] == True:
				trainable_params += summary[layer]["nb_params"]
		print_fn(line_new)

	# assume 4 bytes/number (float on cuda).
	total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
	total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
	total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
	total_size = total_params_size + total_output_size + total_input_size

	print_fn("================================================================")
	print_fn("Total params: {0:,}".format(total_params))
	print_fn("Trainable params: {0:,}".format(trainable_params))
	print_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
	print_fn("----------------------------------------------------------------")
	print_fn("Input size (MB): %0.2f" % total_input_size)
	print_fn("Forward/backward pass size (MB): %0.2f" % total_output_size)
	print_fn("Params size (MB): %0.2f" % total_params_size)
	print_fn("Estimated Total Size (MB): %0.2f" % total_size)
	print_fn("----------------------------------------------------------------")

class Dataset_DCASE2019_t1(data.Dataset):
	#def __init__(self, list_IDs, labels, nb_time, base_dir):
	def __init__(self, lines, d_class_ans, nb_samp, cut, base_dir):
		'''
		self.lines		: list of strings 
		'''
		self.lines = lines 
		self.d_class_ans = d_class_ans 
		self.base_dir = base_dir
		self.nb_samp = nb_samp
		self.cut = cut

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, index):
		k = self.lines[index]
		X = np.load(self.base_dir+k+'.npy')
		y = self.d_class_ans[k.split('-')[0]]

		if self.cut:
			nb_samp = X.shape[1]
			start_idx = np.random.randint(low = 0, high = nb_samp - self.nb_samp)
			X = X[:, start_idx:start_idx+self.nb_samp]
		else: X = X[:, :479999]
		X *= 32000
		return X, y

def split_dcase2019_fold(fold_scp, lines):
	fold_lines = open(fold_scp, 'r').readlines()
	dev_lines = []
	val_lines = []

	fold_list = []
	for line in fold_lines[1:]:
		fold_list.append(line.strip().split('\t')[0].split('/')[1].split('.')[0])
		
	for line in lines:
		if line in fold_list:
			dev_lines.append(line)
		else:
			val_lines.append(line)

	return dev_lines, val_lines

if __name__ == '__main__':
	#load yaml file & set comet_ml config
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)

	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda' if cuda else 'cpu')

	#get DB list
	lines = get_utt_list(parser['DB']+'wave_np')

	#get label dictionary
	d_class_ans, l_class_ans = pk.load(open(parser['DB']+parser['dir_label_dic'], 'rb'))

	#split trnset and devset
	trn_lines, dev_lines = split_dcase2019_fold(fold_scp = parser['DB']+parser['fold_scp'], lines = lines)
	print(len(trn_lines), len(dev_lines))
	del lines

	#define dataset generators
	devset = Dataset_DCASE2019_t1(lines = dev_lines,
		d_class_ans = d_class_ans,
		nb_samp = 0,
		cut = False,
		base_dir = parser['DB']+parser['wav_dir'])
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = False,
		num_workers = parser['nb_proc_db'],
		drop_last = False)

	#load model
	model = raw_CNN_c(parser['model']).to(device)
	model.load_state_dict(torch.load(parser['weight_dir']))
	model.eval()
	SVM = pk.load(open(parser['svm_dir'], 'rb'))
	print(SVM[1], 'class-wise accuracy')
	SVM = SVM[0]

	with torch.set_grad_enabled(False):
		embeddings_dev = []
		data_y_dev = []
		with tqdm(total = len(devset_gen), ncols = 70) as pbar:
			for m_batch, m_label in devset_gen:
				m_batch = m_batch.to(device)
				code, _ = model(m_batch)
				m_label = list(m_label.numpy())
				embeddings_dev.extend(list(code.cpu().numpy())) #>>> (16, 64?)
				data_y_dev.extend(m_label)
				pbar.update(1)
				pbar.set_description('extracting embeddings..')
		embeddings_dev = np.asarray(embeddings_dev, dtype = np.float32)
		print(embeddings_dev.shape)
	score_list = SVM.predict(embeddings_dev)
	print(score_list.shape)
	conf_mat = confusion_matrix(y_true = data_y_dev, y_pred = score_list)
	print(conf_mat)

	conf_mat2 = deepcopy(conf_mat)
	conf_mat2 += conf_mat.T
	rank_dic = {}
	rank_dic_str = {}
	idx = 0
	for i in range(10):
		for j in range(10):
			if i >= j: continue
			rank_dic[conf_mat2[i][j]] = '%d-%d'%(i, j)
			rank_dic_str[conf_mat2[i][j]] = '%s-%s'%(l_class_ans[i], l_class_ans[j])
			idx += 1
			print(idx)

	rank = sorted(rank_dic.keys(), reverse = True)
	rank_classes = []
	rank_classes_str = []
	for c in rank:
		rank_classes.append(rank_dic[c])
		rank_classes_str.append(rank_dic_str[c])
	print(rank_classes)
	print(rank_classes_str)
	print(l_class_ans)
	pk.dump({'rank_classes': rank_classes,
		'rank_classes_str': rank_classes_str,
		'conf_mat': conf_mat}, open(parser['save_dir'], 'wb'))
	

	












