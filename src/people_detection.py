#!/usr/bin/env python


import rospy

import cv2

from cob_perception_msgs.msg import Detection, DetectionArray, Rect

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf 
import time

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec,cm
from matplotlib import pyplot as plt

import numpy as np
import PIL
#from PIL import Image

from deeplab_video4 import DeepLabModel

class PeopleDetectionNode(object):


	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self,MODEL):
		super(PeopleDetectionNode, self).__init__()

		#init the node 
		rospy.init_node('people_detection', anonymous=True)
		self.model = MODEL

		self._bridge = CvBridge()

		

		print('+'*40)

		##init something


		


		# Advertise the result
		self.pub_array = rospy.Publisher('/people_detection_array', DetectionArray , queue_size=1)
		self.pub = rospy.Publisher('/people_detection', Image , queue_size=1)
		self.pub_mask = rospy.Publisher('people_mask', Image , queue_size=1)
		#self.pub = rospy.Publisher('people_detection', Image , queue_size=2)

		#get image from camera
		self.frame = rospy.Subscriber('camera/rgb/image_rect_color',\
			Image, self.people_detection_callback,queue_size=1,buff_size=2**24)
		#cv_depth = self._bridge.imgmsg_to_cv2(self.frame, desired_encoding="passthrough")



		# callback the function
		#self.people_detection_callback(self.frame)

		rospy.spin()

	def read_from_video(self):

		capture = cv2.VideoCapture(0)

		while True:
			ret, frame =capture.read()

			if frame is  not None :
				image_message = self._bridge.cv2_to_imgmsg(frame,'bgr8')

				self.people_detection_callback(image_message)
			else:
				break

		capture.release()
		cv2.destoryAllWindows()
		print(' I HAVE GOT Viedo')

		self.shutdown

		return frame

	def run_visualization(self,frame):
 		"""Inferences DeepLab model and visualizes result."""
		original_im = PIL.Image.fromarray(frame)
		resized_im, seg_map = self.model.run(original_im)

		#image = apply_mask(resized_im, seg_map)
		#seg_image = label_to_color_image(seg_map).astype(np.uint8)
		#image =seg_image
		#cv2.imshow('video',)


		mask = np.zeros([len(seg_map),len(seg_map[0])])
		mask = mask.astype('int8')
		for i in range(len(seg_map)):
			for j in range(len(seg_map[0])):
				if seg_map[i][j] == 15 :
					mask[i][j] = 255
					#print(b)

		return resized_im,mask 

	def shutdown(self):
		"""
		Shuts down the node
		"""
		rospy.signal_shutdown("See ya!")


	def people_detection_callback(self, data):

		cv_image = self._bridge.imgmsg_to_cv2(data,'bgr8')

		image, mask = self.run_visualization(cv_image)
		dim = (480,640)
		image = np.array(image)
		mask = np.array(mask,dtype='uint8')
		res_image = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
		res_mask = cv2.resize(mask,dim,interpolation = cv2.INTER_AREA)

		print('image_shape{}'.format(res_image.shape))
		print('mask_shape {}'.format(res_mask.shape))
		
		res = cv2.bitwise_and(res_image,res_image, mask= res_mask)
		#res_resized = cv2.resize(res,dim,interpolation = cv2.INTER_AREA)
		

		msg_im = self._bridge.cv2_to_imgmsg(res, encoding='passthrough')
		mask_im = self._bridge.cv2_to_imgmsg(res_mask, encoding='passthrough')
		#print('msg_im_type{}'.format(type(msg_im)))
		
		self.pub.publish(msg_im)
		self.pub_mask.publish(mask_im)

		print('I PUBLISHED people_detection_topic')











def main():
	LABEL_NAMES = np.asarray([
	    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
	    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
	    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
	])



	MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

	_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
	_MODEL_URLS = {
	    'mobilenetv2_coco_voctrainaug':
	        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
	    'mobilenetv2_coco_voctrainval':
	        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
	    'xception_coco_voctrainaug':
	        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
	    'xception_coco_voctrainval':
	        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
	}
	_TARBALL_NAME = 'deeplab_model.tar.gz'

	model_dir = tempfile.mkdtemp()
	tf.gfile.MakeDirs(model_dir)

	download_path = os.path.join(model_dir, _TARBALL_NAME)
	print('downloading model, this might take a while...')
	urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
	                   download_path)
	print('download completed! loading DeepLab model...')

	MODEL = DeepLabModel(download_path)
	print('model loaded successfully!')

	node = PeopleDetectionNode(MODEL)


if __name__ == '__main__':
	main()










