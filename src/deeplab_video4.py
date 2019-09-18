#compare to deep_video3,in this i resized the res image from (513,384) to (640,480)
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec,cm
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image

import tensorflow as tf

import cv2

import time

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    #print('width = {} '.format(width))
    #print('height ={}'.format(height))




    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    #print('target_size{}'.format(target_size))

    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    #print('resized_image = {}'.format(resized_image))
    #print('seg_map {}'.format(seg_map.shape))
    return resized_image, seg_map


'''
time1 = time.time()

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



def run_visualization(frame):
  """Inferences DeepLab model and visualizes result."""



  original_im = Image.fromarray(frame)
  resized_im, seg_map = MODEL.run(original_im)

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


capture = cv2.VideoCapture(0)
'''

'''
time3 = 0
while True:
  time2 = time.time()
  print('time = {}'.format(time2-time3))
  time3 = time2 
  ret,frame = capture.read()
  
  #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

  image, mask = run_visualization(frame)
  print(image)
  image = np.array(image)
  print(image.shape)
  print(mask.shape)
  dim = (640,480)

  res = cv2.bitwise_and(image,image, mask= mask)
  res_resized = cv2.resize(res,dim,interpolation = cv2.INTER_AREA)
  print('res_resized_shape{}'.format(res_resized.shape))

  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow('video',res)
  #out.write(image)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  
  

capture.release()
cv2.destroyAllWindows()'''


