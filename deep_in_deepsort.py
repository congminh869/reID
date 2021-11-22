import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
import os
import yaml
from libs.extract_feature_deepsort import FEATURE_IMG
from libs.models import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
import torch.nn as nn
import scipy.io
from torchvision import datasets, models, transforms
from get_iou import get_max_iou



def detect_obj(model, stride, names, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
	global check_select_person
	global feature_person

	imgsz = img_size
	high, weight = img_detect.shape[:2]
	check = False
	#####################################
	classify = False
	agnostic_nms = False
	augment = False
	# Set Dataloader
	#vid_path, vid_writer = None, None
	# Get names and colors

	count = 0
	t = time.time()
	#processing images
	'''
	Tiền xử lí ảnh
	'''
	im0 = letterbox(img_detect, img_size)[0]
	im0 = im0[:, :, ::-1].transpose(2, 0, 1)
	im0 = np.ascontiguousarray(im0)
	im0 = torch.from_numpy(im0).to(device)
	im0 = im0.half() if half else im0.float()
	im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
	if im0.ndimension() == 3:
		im0 = im0.unsqueeze(0)
	# Inference
	t1 = time.time()
	pred = model(im0, augment= augment)[0]
	t2 = time.time()
	print('---------------------------------time detect : ', t2 - t1)
	# Apply NMS
	classes = None
	pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
	# Apply Classifier
	if classify:
		pred = apply_classifier(pred, modelc, im0, img_ocr)
	gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
	points = []
	if len(pred[0]):
		check = True
		pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
		for c in pred[0][:, -1].unique():
			n = (pred[0][:, -1] == c).sum()  # detections per class
		for box in pred[0]:
			c1 = (int(box[0]), int(box[1]))
			c2 = (int(box[2]), int(box[3]))
			x1, y1 = c1
			x2, y2 = c2
			acc = round(float(box[4])*100,2)
			cls = int(box[5])
			conf = box[4].item()
			label = names[cls]#
			# print(label)
			img_crop = img_detect[y1:y2, x1:x2]
			if label == 'person':
				points.append(np.array([x1,y1,x2,y2]))
				# cv2.rectangle(img_detect, c1, c2, (0,0,255), 2)
				if check_select_person == False:
					# print('check_select_person = False')
					t3 = time.time()
					id_person = obj_track.track(img_crop)
					t4 = time.time()
					print('time tracking   : ', t4 - t3)
					if id_person:
						cv2.rectangle(img_detect, c1, c2, (0,0,255), 2)
				# cv2.imshow('frame ', img_detect)

	if check_select_person:
		# print('bat dau chon nguoi')
		point_person = Select_person(img_detect, np.array(points))
		x1_pp, y1_pp, x2_pp, y2_pp = point_person
		feature_person = img_detect[y1_pp:y2_pp, x1_pp:x2_pp]
		obj_track.update(feature_person )	
	# print('Processing time %0.3f s'%(time.time()-t))
	return img_detect

def Select_person(img_detect, points):
	global check_select_person
	global names
	for point in points:
		x1,y1,x2,y2 = point
		cv2.rectangle(img_detect, (x1,y1), (x2,y2), (0,0,255), 2)
	bboxes = []
	# while True:
	# 	bbox = cv2.selectROI('MultiTracker', frame)
	# 	bboxes.append(bbox)
	# 	print("Press q to quit selecting boxes and start tracking")
	# 	print("Press any other key to select next object")
	# 	k = cv2.waitKey(0) & 0xFF
	# 	if (k == 113):  # q is pressed
	# 		cv2.destroyAllWindows()
	# 		break

	bboxes = [(186, 204, 37, 115)]

	print('Selected bounding boxes {}'.format(bboxes))

	for gt_bbox_xywh in bboxes:
		gt_bbox = (gt_bbox_xywh[0], gt_bbox_xywh[1], gt_bbox_xywh[0] + gt_bbox_xywh[2], gt_bbox_xywh[1] + gt_bbox_xywh[3])
		print('gt_bbox : ', gt_bbox)

		iou, iou_max, nmax = get_max_iou(points, gt_bbox)

		point_person =points[iou.argmax(axis=0)]
		print('point_person : ', point_person)
		print('points : ',points)
	check_select_person = False
	return point_person


if __name__ == '__main__':
	check_select_person = True
	index_ids = None
	feature_person = None


	use_gpu = torch.cuda.is_available()
	print('use_gpu : ',use_gpu)
	img_size = 640
	conf_thres = 0.25
	iou_thres = 0.45
	device = ''
	update = True
	# Load model yolo
	print('=================Loading models=================')
	t1 = time.time()
	set_logging()
	device = select_device(device)
	half = device.type != 'cpu'  # half precision only supported on CUDA
	# Load model nhan dien container
	t1 = time.time()
	weights = './model/yolov5s.pt'
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	names = model.module.names if hasattr(model, 'module') else model.names
	if half:
		model.half()
	t2 = time.time()
	print('time load model yolo : ', t2-t1)


	#Load model tracking person





	# Load feature from data
	print('=================Loading data=================')
	transImage = transforms.Compose([transforms.ToPILImage(),
		transforms.Resize((256, 128), interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	#load tracking 
	obj_track = FEATURE_IMG()
	
	# MAIN
	cap = cv2.VideoCapture('./data/test.avi')
	width, height = (0, 0)
	if (cap.isOpened() == False): 
		print("Error reading video file")
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	size = (frame_width, frame_height)
	result = cv2.VideoWriter('filename_1thread.avi',\
							cv2.VideoWriter_fourcc(*'MJPG'),10, size)
	print('Starting convert')
	count = 0
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			# Display the resulting frame
			t7 = time.time()
			frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 320)
			t8 = time.time()
			print('time total ----------------------------------------------------------------------',t8-t7)
			# print(frame.shape[:2])
			count+= 1
			if count % 100==0:
				print(count)
			result.write(frame)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		# Break the loop
		else:
			break
	cap.release()
	out.release()
	cv2.destroyAllWindows()