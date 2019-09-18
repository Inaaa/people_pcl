cc_people_pointcloud

Mask rcnn, tensorrt,tensorflow This package demonstrated people semantic segmentation using ROS.

Tech

This repo uses a number of open source projects to work properly:

[Tensorflow] [Tensorflow-Object Detection API] [Tensorflow Hub] [ROS] [Numpy] [cob_perception_common] https://github.com/ipa-rmb/cob_perception_common.git [ros_people_object_detection_tensorflow] https://github.com/cagbal/ros_people_object_detection_tensorflow.git

Installation and Running

First, tensorflow-gpu should be installed on your system. In this project I use tensorflow-gpu 1.13 and cuda 10 .

Then, $ cd && mkdir -p catkin_ws/src && cd catkin_ws $ catkin_make && cd src $ git clone --recursive https://github.com/Inaaa/ros_people_object_detection_tensorflow.git,  and then git checkout cgo-cl
Second change the model_name parameter of launch/cob_people_object_detection_tensoflow_params.yaml into mask_rcnn_inception_v2_coco_tensorrt.

Here you should at first run the ros_people_object_detection_tensorflow. Furthermore, $ roslaunch cc_people_pointcloud rgbdpoints.launch
