import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import yaml
import math
from libs.models import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
import cv2
def load_network(network, save_path):
	network.load_state_dict(torch.load(save_path))
	return network
class FEATURE_IMG():
	def __init__(self, model):
		self.model = model		
		self.transImage = transforms.Compose([transforms.ToPILImage(),
			transforms.Resize((256, 128), interpolation=3),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		self.person_select = torch.FloatTensor()
	def img_feature(self, img):
		im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = self.transImage(im_rgb).unsqueeze(0)
		feature = torch.FloatTensor()
		n, c, h, w = img.size()
		if torch.cuda.is_available():
			ff = torch.FloatTensor(n, 512).zero_().cuda()
		else:
			ff = torch.FloatTensor(n, 512).zero_()
		for i in range(2):
			if (i == 1):  # Flip the image
				inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
				img = img.index_select(3, inv_idx)
			if torch.cuda.is_available():
				input_img = Variable(img.cuda())
			else:
				input_img = Variable(img.cpu())
			t5 = time.time()
			outputs = self.model(input_img)
			t6 = time.time()
			# print('time tracking 1 object: ', t6-t5)
			ff += outputs
		fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
		ff = ff.div(fnorm.expand_as(ff))
		feature = torch.cat((feature, ff.data.cpu()), 0)
		return feature
	def track(self, img):
		img_c = self.img_feature(img)
		# img_ps = self.img_feature(person_select)
		# a = self.person_selectview(-1, 1)
		score = torch.mm(img_c, self.person_select.view(-1, 1))
		# print('score : ', score)
		if score.item() > 0.6:
			# self.person_select = img_c
			# print('True : ', score)
			return True
		return False
	def update(self, img_query):
		self.person_select = self.img_feature(img_query)
