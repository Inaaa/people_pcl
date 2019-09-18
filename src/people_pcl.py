#!/usr/bin/env python

import rospy

import cv2

import tensorflow as tf 

import time

import tarfile

import tempfile

from six.moves import urllib

import numpy as np

import PIL

import rospy

import numpy as np

import message_filters

import math

import struct

import os

from io import BytesIO

from cob_perception_msgs.msg import Detection, DetectionArray, Rect

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs import point_cloud2

from sensor_msgs.msg import Image ,PointCloud2, PointField

from deeplab_video4 import DeepLabModel

from std_msgs.msg import Header

import sensor_msgs.point_cloud2 as pc2


class PeoplePCL(object):

	def __init__(self,model):
		super(PeoplePCL, self).__init__()

		# init the node
		rospy.init_node('people_pcl', anonymous=True)

		self.model = model
		self._bridge = CvBridge()

		# Subscribe the mask_image, depth_image und rgb_image
		sub_depth = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect', Image)
		sub_rgb = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)
		#print('sub_depth{}'.format(sub_depth))


		# Advertise the result
		self.pub_image = rospy.Publisher('/people_detection', Image , queue_size=1)
		self.pub_mask = rospy.Publisher('/people_mask', Image , queue_size=1)
		self.pub = rospy.Publisher('people_pcl', PointCloud2, queue_size=1)
		self.pub_depth_mask = rospy.Publisher('people_pcl_mask', Image, queue_size=1)

		# Create the message filter
		ts = message_filters.ApproximateTimeSynchronizer( \
		    [sub_depth,sub_rgb], \
		    2, \
		    0.5)

		ts.registerCallback(self.people_callback)

		self.f = 587.8
		self.cx = 351
		self.cy = 227

		# spin
		rospy.spin()

	def shutdown(self):
		"""
		shuts down the node
		"""
		rospy.signal_shutdown("see you later")

	def run_visualization(self,frame):
 		"""Inferences DeepLab model and visualizes result."""
		original_im = PIL.Image.fromarray(frame)
		resized_im, seg_map = self.model.run(original_im)

		#get the mask only for person
		mask = np.zeros([len(seg_map),len(seg_map[0])])
		#print('mask_size{}'.format(mask.shape))
		mask = mask.astype('int8')
		for i in range(len(seg_map)):
			for j in range(len(seg_map[0])):
				if seg_map[i][j] == 15 :
					mask[i][j] = 1

		return resized_im,mask 


	def people_callback(self,depth,data):
		cv_image = self._bridge.imgmsg_to_cv2(data,'bgr8')
		cv_depth = self._bridge.imgmsg_to_cv2(depth,"passthrough")
		cv_depth_array =np.array(cv_depth)
		#print(cv_depth_array.dtype)
		#print(np.unique(cv_depth_array,return_counts=True))

		image, mask_org = self.run_visualization(cv_image)
		dim = (640,480)
		image = np.array(image)

		mask = np.array(mask_org,dtype='uint8')
		res_image = cv2.resize(image,dim)
		res_mask = cv2.resize(mask,dim,interpolation = cv2.INTER_AREA)

		#print('image_shape{}'.format(res_image.shape))
		#print('mask_shape {}'.format(res_mask.shape))
		
		res = cv2.bitwise_and(res_image,res_image, mask= res_mask)

		msg_im = self._bridge.cv2_to_imgmsg(res, encoding='passthrough')
		mask_im = self._bridge.cv2_to_imgmsg(res_mask, encoding='passthrough')

		self.pub_image.publish(msg_im)
		self.pub_mask.publish(mask_im)

		mask_depth = cv2.bitwise_and(cv_depth_array,cv_depth_array,mask=res_mask)
		#print(np.unique(mask_depth,return_counts=True))

		#print(np.unique(res))

		mask_depth_im = self._bridge.cv2_to_imgmsg(mask_depth, encoding='passthrough')
		self.pub_depth_mask.publish(mask_depth_im)
		pcl=self.pointcloud_ge(res, mask_depth)
		self.pub.publish(pcl)
		print('Good generate pcl')


	def pointcloud_ge(self, image, mask_depth):
		'''
		msg = masked rgb image
		depth = depth image

		'''
		#convert massage to numpy array			

		print(image.shape)
		print(mask_depth.shape)
		pointcloud = []
		time0 = time.time()
		for u in range (image.shape[0]):
		    for v in range(image.shape[1]):
				Z = mask_depth.item(u,v)

				if Z == 0.0 or Z > 1.5:
					pass
				else:
					color = image[u,v]
					X = (u-self.cx)*Z/self.f
					Y = (v-self.cy)*Z/self.f
					x = np.float(X)
					y = np.float(Y)
					z = np.float(Z)

					r = color[0]
					g = color[1]
					b = color[2]
					a = 255

					rgb = struct.unpack('I',struct.pack('BBBB',r,g,b,a))[0]

					points =[x,y,z,rgb]

					pointcloud.append(points)
		
		time1 =time.time()
		print('TIME {}'.format(time1-time0))
		fields = [PointField('x', 0, PointField.FLOAT32, 1),
					PointField('y', 4, PointField.FLOAT32, 1),
					PointField('z', 8, PointField.FLOAT32, 1),
					PointField('rgb', 12, PointField.UINT32, 1),
					]

		header =Header()
		header.frame_id = "camera_rgb_optical_frame"

		point_generate = point_cloud2.create_cloud(header, fields, pointcloud)
		point_generate.header.stamp = rospy.Time.now()

		return point_generate





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

	node = PeoplePCL(MODEL)


if __name__ == '__main__':
	main()

