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
from libs.extract_feature_minh import FEATURE_IMG, load_network
from libs.models import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
import torch.nn as nn
import scipy.io
from torchvision import datasets, models, transforms
from get_iou import get_max_iou

import zmq
import socket
import pickle
import base64
from threading import Thread,Lock
import threading
mutex = Lock()

from control_dc import ratio #sudo chmod 666 /dev/ttyUSB0
ratio(0)

def camera_ratio(x,y,high, weight):
	a = 50
	thres_min_x = weight/2 - a
	thres_max_x = weight/2 + a
	thres_min_y = high/2 - a
	thres_max_y = high/2 + a

	if x>=thres_max_x and y>=thres_min_y and y<=thres_max_y: # right
		return 2
	elif x>=thres_min_x and x<=thres_max_x and y<=thres_min_y: #up
		return 3
	elif x<=thres_min_x and y>=thres_min_y and y<=thres_max_y:#left
		return 1
	elif x>=thres_min_x and x<=thres_max_x and y>=thres_max_y:#down
		return 4
	elif x<=thres_min_x and y>=thres_max_y:	#left down
		return 6
	elif x>=thres_max_x and y>=thres_max_y:	#down right
		return 8
	elif x>=thres_max_x and y<=thres_min_y:	#right up
		return 7
	elif x<=thres_min_x and y<=thres_min_y:#up left
		return 5
	else:
		return 9
	# elif x>=thres_min_x and x<=thres_max_x and y>=thres_min_y and y<=thres_max_y:#stop
	# 	return 9

def detect_obj(model, stride, names, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
	global check_select_person
	global feature_person

	imgsz = img_size
	high, weight = img_detect.shape[:2]
	# print('********************')
	# print(high, weight)
	# print('********************')
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
	# print('time detect : ', t2 - t1)
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
			x_center = int((x1+x2)/2)
			y_center = int((y1+y2)/2)
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
					# print('time tracking   : ', t4 - t3)
					if id_person:
						signal = camera_ratio(x_center,y_center, high, weight)
						text_signal = ['left', 'right', 'up', 'down', 'left+up', 'left+down', 'right+up', 'right+down', 'stop']
						cv2.rectangle(img_detect, c1, c2, (0,0,255), 2)
						cv2.rectangle(img_detect, (x_center,y_center), (x_center+2,y_center+2), (0,0,255), 2)
						ratio(signal)
						text_ratio = '('+str(x_center)+ ',' +str(y_center) + ') ' + text_signal[signal-1]
						cv2.putText(img_detect, text_ratio , (x_center,y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
						cv2.putText(img_detect, '('+str(high)+','+str(weight)+')' , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	if check_select_person:
		print('check_select_person')
		point_person = Select_person(img_detect, np.array(points))
		x1_pp, y1_pp, x2_pp, y2_pp = point_person
		feature_person = img_detect[y1_pp:y2_pp, x1_pp:x2_pp]
		print('update frame', point_person)
		obj_track.update(feature_person )	
	return img_detect

def thread_socket():
	print('==================================start threading socket==================================')
	global rec_done,_close,bboxesT,clinet_connect,check_select_person
	_close = False
	rec_done = False
	clinet_connect = False

	# frame = img_detect
	HOST = '0.0.0.0'
	PORT = 8000
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)      
	s.bind((HOST, PORT))
	s.listen()

	print('wait connect socket from client')
	count_c = 0
	check_close = True
	while True:
		print('count_c : ',count_c)
		if check_close:
			conn, addr = s.accept()
			check_close = False
		print('+++++++++reconnected {} connected. +++++++++++'.format(addr))
		mutex:acquire()
		check_select_person = True
		mutex:release()
		clinet_connect = True
		conn.sendall(b'chon anh')
		time.sleep(5)
		print('line ******1******* ')
		data_recv = conn.recv(1024)
		print(data_recv)
		print('line ******2******* ')
		print('Start listening...')

		print('data_recv ',data_recv)
		if data_recv == b'close':
			print('------------Gui------close')
			check_select_person = True
			check_close = True
			# conn.sendall('close'.encode())
			check_close = True
			# break
		# elif data_recv == b'done':
		else:
			print('-----------nhan kich thuoc-------close')
			data = pickle.loads(data_recv)
			print(data)
			print(type(data))
			# mutex:release()
			conn.sendall('close'.encode())
			print('continue----thread')
			bboxesT = data 
			rec_done = True
		print('-----------doi offff++++++++++++++++++++++++++++++')
		count_c +=1

	print('da dong socket')
        
	# s.close()
	# conn.close()

def Select_person(img_detect, points):
	global check_select_person, clinet_connect,bboxesT,data1
	global rec_done,_close
	global names
	for point in points:
		x1,y1,x2,y2 = point
		cv2.rectangle(img_detect, (x1,y1), (x2,y2), (0,0,255), 2)
	bboxes = []

	check = True
	count = 0
	while True:
		#dung lai cho den khi client connect
		if clinet_connect:
			print('--------------ennable client')
			break
	while True:
		print('-----ennable client-----',clinet_connect)
		# high, weight = img_detect.shape[:2]
		# img_detect = cv2.resize(img_detect, (high/2,weight/2))
		encoded, buffer = cv2.imencode('.jpg', img_detect)
		jpg_as_text = base64.b64encode(buffer)
		footage_socket.send(jpg_as_text)

		k = cv2.waitKey(1) & 0xFF
		count +=1
		print('time.sleep(5) , count = ', count)
		if check:
			time.sleep(2)
		if count >= 3:
			if check:
				while True:
					if _close:
						print('-------turn off-------close')
						k=113
						check=False
						_close = False
						rec_done = False
						break
					# else:
					elif rec_done == True:
						#data = pickle.loads(data_recv)
						bboxes = bboxesT
						print('---Nhan anh deted xong')
						_close = True
						print('continue')
			
			if (k == 113):  # q is pressed
				print('k = ', k)
				break

	# print('Selected bounding boxes {}'.format(bboxes))

	for gt_bbox_xywh in bboxes:
		gt_bbox = (gt_bbox_xywh[0], gt_bbox_xywh[1], gt_bbox_xywh[0] + gt_bbox_xywh[2], gt_bbox_xywh[1] + gt_bbox_xywh[3])
		# print('gt_bbox : ', gt_bbox)

		iou, iou_max, nmax = get_max_iou(points, gt_bbox)

		print('===============')
		print(iou)
		print('===============')

		point_person =points[iou.argmax(axis=0)]
		print('point_person : ', point_person)
		print('points : ',points)
	check_select_person = False
	return point_person


if __name__ == '__main__':
	check_select_person = True
	index_ids = None
	feature_person = None
	clinet_connect = False
	context = zmq.Context()
	footage_socket = context.socket(zmq.PUB)
	footage_socket.connect('tcp://192.168.0.101:5555')


	use_gpu = torch.cuda.is_available()
	print('use_gpu : ',use_gpu)
	img_size = 640
	conf_thres = 0.25
	iou_thres = 0.45
	device = ''
	update = True
	# Load model yolo
	print('=================Loading models yolov5=================')
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
	config_path = './model/ft_ResNet50/opts1.yaml'# path config
	with open(config_path, 'r') as stream:
	  config = yaml.load(stream)

	use_dense = config['use_dense']
	use_NAS = config['use_NAS']
	stride = config['stride']
	nclasses = config['nclasses']
	batchsize = 256
	#which_epoch = opt.which_epoch
	gpu_ids = ['cuda:0']
	ms = [1]
	# set gpu ids
	if len(gpu_ids)>0 and torch.cuda.is_available():
		torch.cuda.set_device(gpu_ids[0])
		cudnn.benchmark = True
	if use_dense:
		# print('use_dense')
		model_structure = ft_net_dense(nclasses)
	elif use_NAS:
		# print('use_NAS')
		model_structure = ft_net_NAS(nclasses)
	else:
		print('ft_ResNet50')
		model_structure = ft_net(nclasses, stride = stride)

		print('=================Loading models ft_ResNet50=================')
	t3 = time.time()
	model_track = load_network(model_structure,'./model/ft_ResNet50/net_last_minh.pth')
	t4 = time.time()

	print('time load model ft_ResNet50 : ', t4-t3)
	# Remove the final fc layer and classifier layer
	model_track.classifier.classifier = nn.Sequential()

	# Change to test mode
	model_track = model_track.eval()

	if use_gpu:
		model_track = model_track.cuda()
	print('Loaded models in %0.3f s'%(time.time()-t1))
	# Load feature from data
	transImage = transforms.Compose([transforms.ToPILImage(),
		transforms.Resize((256, 128), interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	#load tracking 
	obj_track = FEATURE_IMG(model_track)

	# MAIN
	cap = cv2.VideoCapture(0)
	width, height = (0, 0)
	if (cap.isOpened() == False): 
		print("Error reading video file")
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	size = (frame_width, frame_height)
	result = cv2.VideoWriter('filename_1thread.avi',\
							cv2.VideoWriter_fourcc(*'MJPG'),10, size)


	#Starting 
	print('Starting thread')
	thread1 = threading.Thread(name='thread_socket', target = thread_socket)
	thread1.start()
	print('Starting convert')
	count = 0
	print('==================================start detect object==================================')
	try:
		while(cap.isOpened()):
			# Capture frame-by-frame
			ret, frame = cap.read()
			if ret == True:
				# Display the resulting frame
				t7 = time.time()
				# frame = cv2.resize(frame, (640,640)) 
				frame = detect_obj(model, stride, names, img_detect = frame, iou_thres = 0.4, conf_thres = 0.5, img_size = 320)
				frame_wwrite = frame
				if check_select_person == False:
					high, weight = frame.shape[:2]
					w = int(weight/2)
					h = int(high/2)
					frame = cv2.resize(frame, (w,h))
					encoded, buffer = cv2.imencode('.jpg', frame)
					#cv2.imshow('abc', )
					jpg_as_text = base64.b64encode(buffer)
					footage_socket.send(jpg_as_text)
				t8 = time.time()
				# print('time total ----------------------------------------------------------------------',t8-t7)
				# print(frame.shape[:2])
				count+= 1
				if count % 100==0:
					print(count)
				result.write(frame_wwrite)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			# Break the loop
			else:
				break
	except KeyboardInterrupt:
		cv2.destroyAllWindows()
		cap.release()
		result.release()
	# cv2.destroyAllWindows()
