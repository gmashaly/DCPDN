import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable


from myutils.vgg16 import Vgg16

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
	img = Image.open(filename).convert('RGB')
	if size is not None:
		if keep_asp:
			size2 = int(size * 1.0 / img.size[0] * img.size[1])
			img = img.resize((size, size2), Image.ANTIALIAS)
		else:
			img = img.resize((size, size), Image.ANTIALIAS)

	elif scale is not None:
		img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
	img = np.array(img).transpose(2, 0, 1)
	img = torch.from_numpy(img).float()
	return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
	(b, g, r) = torch.chunk(tensor, 3)
	tensor = torch.cat((r, g, b))
	tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def subtract_imagenet_mean_batch(batch):
	"""Subtract ImageNet mean pixel-wise from a BGR image."""
	tensortype = type(batch.data)
	mean = tensortype(batch.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	return batch - Variable(mean)


def add_imagenet_mean_batch(batch):
	"""Add ImageNet mean pixel-wise from a BGR image."""
	tensortype = type(batch.data)
	mean = tensortype(batch.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	return batch + Variable(mean)

def imagenet_clamp_batch(batch, low, high):
	batch[:,0,:,:].data.clamp_(low-103.939, high-103.939)
	batch[:,1,:,:].data.clamp_(low-116.779, high-116.779)
	batch[:,2,:,:].data.clamp_(low-123.680, high-123.680)


def preprocess_batch(batch):
	batch = batch.transpose(0, 1)
	(r, g, b) = torch.chunk(batch, 3)
	batch = torch.cat((b, g, r))
	batch = batch.transpose(0, 1)
	return batch


def init_vgg16(model_folder):
	from torch.utils.serialization import load_lua
	"""load the vgg16 model feature"""
	if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
		if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
			os.system(
				'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
		vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
		vgg = Vgg16()
		for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
			dst.data[:] = src
		torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))



def update_mddel_kwys(state_dict):
	import re
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		dot_key = re.findall(r"_est.layer\d.layer\d.conv.", k) or re.findall(r"_est.[d]*layer[final]*\d.[d]*layer\d.bn.",k) or re.findall(r"_est.dlayer\d.dlayer\d.tconv.", k)
		dot_key = dot_key or re.findall(r"_est.[d]*layer[final]*.[d]*layer\d.conv.", k)
		dot_key = dot_key or re.findall(r"tran_dense.dense_block\d.denselayer\d{1,2}.norm.", k)
		name = k
		if dot_key:
			name = k.replace('.conv.', '_conv.')
			name = name.replace('.bn.', '_bn.')
			name = name.replace('.tconv.', '_tconv.')
			name = name.replace('.norm.', '.norm')
		dot_key = re.findall(r"tran_dense.dense_block\d.denselayer\d{1,2}.conv.", k)
		if dot_key:
			name = name.replace('.conv.', '.conv')

		new_state_dict[name] = v

	return new_state_dict