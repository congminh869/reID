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

from libs.deep.feature_extractor import Extractor

class FEATURE_IMG():
	def __init__(self):	
		self.transImage = transforms.Compose([transforms.ToPILImage(),
			transforms.Resize((256, 128), interpolation=3),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		self.person_select = torch.FloatTensor()

		model_path = './libs/deep/checkpoint/ckpt.t7'
		if torch.cuda.is_available():
			use_gpu = True
		else:
			use_gpu = False
		self.extractor = Extractor(model_path, use_cuda=use_gpu)

	def img_feature(self, img):
		features = self.extractor(img)
		return features

	def track(self, img):
		img_c = self.img_feature(img)
		# img_ps = self.img_feature(person_select)
		# a = self.person_selectview(-1, 1)
		score = torch.mm(img_c, self.person_select.view(-1, 1))
		print('score : ', score)
		if score.item() > 0.97:
			# self.person_select = img_c
			# print('True : ', score)
			return True
		return False
	def update(self, img_query):
		self.person_select = self.img_feature(img_query)