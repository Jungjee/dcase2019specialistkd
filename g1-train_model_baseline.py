from comet_ml import Experiment
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.svm import SVC

import os
import yaml
import struct
import pickle as pk
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

from g_model_CNN_C import raw_CNN_c

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
		self.d_ark_files = {}


	def __len__(self):
		return len(self.lines)


	def __getitem__(self, index):
		line = self.lines[index]
		k, f, p = line.strip().split(' ')
		y = self.d_class_ans[k.split('-')[0]]
		p = int(p)

		if not f in self.d_ark_files:
			f_tmp = '/'.join([self.base_dir, f])
			self.d_ark_files[f] = open(f_tmp, 'rb')
		self.d_ark_files[f].seek(p)

		l = struct.unpack('i', self.d_ark_files[f].read(4))[0]
		X = struct.unpack('%df'%(479520*2), self.d_ark_files[f].read((479520*2) *4))
		#X = struct.unpack('%df'%(l*2), self.d_ark_files[f].read((l*2) *4))
		X = np.asarray(X, dtype = np.float32).reshape(-1,2).T

		if self.cut:
			nb_samp = X.shape[1]
			start_idx = np.random.randint(low = 0, high = nb_samp - self.nb_samp)
			X = X[:, start_idx:start_idx+self.nb_samp]
		X *= 32000
		return X, y
	

def make_labeldic(lines):
	idx = 0
	dic_label = {}
	list_label = []
	for line in lines:
		k, f, p = line.strip().split(' ')
		label = k.split('-')[0]
		if label not in dic_label:
			dic_label[label] = idx
			list_label.append(label)
			idx += 1
	return (dic_label, list_label)


def split_dcase2019_fold(fold_scp, lines):
	fold_lines = open(fold_scp, 'r').readlines()
	dev_lines = []
	val_lines = []

	fold_list = []
	for line in fold_lines[1:]:
		fold_list.append(line.strip().split('\t')[0].split('/')[1])
		
	for line in lines:
		key = line.strip().split(' ')[0]
		if key in fold_list:
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
	experiment = Experiment(api_key="9CueLwB3ujfFlhdD9Z2VpKKaq",
		project_name="torch_dcase2019", workspace="jungjee",
		auto_output_logging = 'simple',
		disabled = bool(parser['comet_disable']))
	experiment.set_name(parser['name'])
	
	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

	#get label dictionary
	with open(parser['dir_scp'], 'r') as f:
		lines = f.readlines()
	if bool(parser['make_label_dic']):
		d_class_ans, l_class_ans  = make_labeldic(lines)
		pk.dump([d_class_ans, l_class_ans], open(parser['dir_label_dic'], 'wb'))
	else:
		d_class_ans, l_class_ans = pk.load(open(parser['dir_label_dic'], 'rb'))
	print(d_class_ans)

	#split trnset and devset
	trn_lines, dev_lines = split_dcase2019_fold(fold_scp = parser['fold_scp'], lines = lines)
	del lines
	if bool(parser['comet_disable']):
		np.random.shuffle(trn_lines)
		np.random.shuffle(dev_lines)
		trn_lines = trn_lines[:1000]
		dev_lines = dev_lines[:1000]

	#define dataset generators
	trnset = Dataset_DCASE2019_t1(lines = trn_lines,
		d_class_ans = d_class_ans,
		nb_samp = parser['nb_samp'],
		cut = True,
		base_dir = parser['DB'])
	trnset_gen = data.DataLoader(trnset,
		batch_size = parser['batch_size'],
		shuffle = True,
		num_workers = parser['nb_proc_db'],
		drop_last = True)
	devset = Dataset_DCASE2019_t1(lines = dev_lines,
		d_class_ans = d_class_ans,
		nb_samp = 0,
		cut = False,
		base_dir = parser['DB'])
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = False,
		num_workers = parser['nb_proc_db'],
		drop_last = False)

	#set save directory
	save_dir = parser['save_dir'] + parser['name'] + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir  + 'results/'):
		os.makedirs(save_dir + 'results/')
	if not os.path.exists(save_dir  + 'weights/'):
		os.makedirs(save_dir + 'weights/')
	if not os.path.exists(save_dir  + 'svm/'):
		os.makedirs(save_dir + 'svm/')
	
	#log experiment parameters to local and comet_ml server
	#to local
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in parser.items():
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.close()

	#to comet server
	experiment.log_parameters(parser)
	experiment.log_parameters(parser['model'])

	#define model
	model = raw_CNN_c(parser['model']).to(device)

	#log model summary to file
	with open(save_dir + 'summary.txt', 'w+') as f_summary:
		#summ = summary(model, input_size = (parser['model']['in_channels'], parser['nb_time'], parser['feat_dim'])) # configure input_size as (channels, H, W)
		summary(model,
			input_size = (parser['model']['in_channels'], parser['nb_samp']),
			print_fn=lambda x: f_summary.write(x + '\n')) 

	if len(parser['gpu_idx']) > 1:
		model = nn.DataParallel(model, device_ids = parser['gpu_idx'])

	#set ojbective funtions
	criterion = nn.CrossEntropyLoss()
	c_obj_fn = CenterLoss(num_classes = parser['model']['nb_classes'],
		feat_dim = parser['model']['nb_fc_node'],
		device = device)

	#set optimizer
	params = list(model.parameters()) + list(c_obj_fn.parameters())
	if parser['optimizer'].lower() == 'sgd':
		optimizer = torch.optim.SGD(params,
			lr = parser['lr'],
			momentum = parser['opt_mom'],
			weight_decay = parser['wd'],
			nesterov = bool(parser['nesterov']))

	elif parser['optimizer'].lower() == 'adam':
		#optimizer = torch.optim.Adam(params,
		optimizer = torch.optim.Adam(model.parameters(),
			lr = parser['lr'],
			weight_decay = parser['wd'],
			amsgrad = bool(parser['amsgrad']))
	'''
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
		milestones = parser['lrdec_milestones'],
		gamma = parser['lrdec'])
	'''
	##########################################
	#train/val################################
	##########################################
	best_acc = 0.
	f_acc = open(save_dir + 'accs.txt', 'a', buffering = 1)
	for epoch in tqdm(range(parser['epoch'])):
		#train phase
		model.train()
		mixup = True if epoch > parser['mixup_start'] else False
		with tqdm(total = len(trnset_gen), ncols = 70) as pbar:
			for m_batch, m_label in trnset_gen:
				#continue #temporary
				m_batch, m_label = m_batch.to(device), m_label.to(device)
				
				#mixup data if met condition
				if mixup:
					m_batch, m_label_a, m_label_b, lam = mixup_data(m_batch, m_label,
						alpha = parser['mixup_alpha'],
						use_cuda = True)
					m_batch, m_label_a, m_label_b = map(torch.autograd.Variable, [m_batch, m_label_a, m_label_b])
					code, output = model(m_batch)
					cce_loss = mixup_criterion(criterion, output, m_label_a, m_label_b, lam)
					c_loss = mixup_criterion(c_obj_fn, code, m_label_a, m_label_b, lam)
				else:
					code, output = model(m_batch)
					cce_loss = criterion(output, m_label)
					c_loss = c_obj_fn(code, m_label)
				loss = cce_loss + (parser['c_loss_weight'] * c_loss)

				#print(loss)
				optimizer.zero_grad()
				loss.backward()
				for param in c_obj_fn.parameters():
					param.grad.data *= (parser['c_loss_lr'] / (parser['c_loss_weight'] * parser['lr']))
				optimizer.step()
				pbar.set_description('epoch: %d loss: %.3f'%(epoch, loss))
				pbar.update(1)
		experiment.log_metric('trn_loss', loss)
		#lr_scheduler.step()

		#validation phase
		model.eval()
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
					pbar.set_description('epoch%d: Extract ValEmbeddings'%(epoch))
					pbar.update(1)
			embeddings_dev = np.asarray(embeddings_dev, dtype = np.float32)
			print(embeddings_dev.shape)

			embeddings_trn = []
			data_y = []
			with tqdm(total = len(trnset_gen), ncols = 70) as pbar:
				for m_batch, m_label in trnset_gen:
					m_batch = m_batch.to(device)
					code, _ = model(m_batch)
					m_label = list(m_label.numpy())
					embeddings_trn.extend(list(code.cpu().numpy())) #>>> (16, 64?)
					data_y.extend(m_label)
					pbar.set_description('epoch%d: Extract TrnEmbeddings'%(epoch))
					pbar.update(1)
			embeddings_trn = np.asarray(embeddings_trn, dtype = np.float32)
			
			SVM_list = []
			acc = []
			classwise_acc = []
			for cov_type in ['rbf', 'sigmoid']:
				score_list = []
		
				SVM_list.append(SVC(kernel=cov_type,
					gamma = 'scale',
					probability = True))
				SVM_list[-1].fit(embeddings_trn, data_y)
		
				num_corr = 0
				num_corr_class = [0]* len(l_class_ans)
				num_predict_class = [0] * len(l_class_ans)
		
				score_list = SVM_list[-1].predict(embeddings_dev)
				
				assert len(score_list) == len(data_y_dev)
				for i in range(embeddings_dev.shape[0]):
					num_predict_class[score_list[i]] += 1
					#print(score_list[i], data_y_dev[i])
					if score_list[i] == data_y_dev[i]:
						#print('cor')
						num_corr += 1
						num_corr_class[data_y_dev[i]] += 1
				acc.append(float(num_corr)/ embeddings_dev.shape[0])
				classwise_acc.append(np.array(num_corr_class) / np.array(num_predict_class))
				print(classwise_acc[-1], acc[-1])
			f_acc.write('%d %f %f\n'%(epoch, float(acc[0]), float(acc[1])))
	
			max_acc = max(acc[0], acc[1])
			experiment.log_metric('val_acc_rbf', acc[0])
			experiment.log_metric('val_acc_sig', acc[1])
			#record best validation model
			if max_acc > best_acc:
				print('New best acc: %f'%float(max_acc))
				best_acc = float(max_acc)
				experiment.log_metric('best_val_acc', best_acc)
				
				#save best model
				if acc[0] > acc[1]:
					pk.dump((SVM_list[0], classwise_acc[0]), open(save_dir + 'svm/best_rbf.pk', 'wb'))
					if len(parser['gpu_idx']) > 1: # multi GPUs
						torch.save(model.module.state_dict(), save_dir +  'weights/best_rbf.pt')
					else: #single GPU
						torch.save(model.state_dict(), save_dir +  'weights/best_rbf.pt')
				else:
					pk.dump((SVM_list[1], classwise_acc[1]), open(save_dir + 'svm/best_sig.pk', 'wb'))
					if len(parser['gpu_idx']) > 1: # multi GPUs
						torch.save(model.module.state_dict(), save_dir +  'weights/best_sig.pt')
					else: #single GPU
						torch.save(model.state_dict(), save_dir +  'weights/best_sig.pt')
				
	f_acc.close()

















