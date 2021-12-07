# TransferLearning

import tensorflow as tf
import pprint
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import numpy as np

# import the models for further classification experiments

from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )
 
# init the models
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.imagenet_utils import decode_predictions

# assign the image path for the classification experiments

filename = 'IMG_0140.jpg'

# load an image in PIL format

original = load_img(filename, target_size=(224, 224))

print('PIL image size',original.size)

plt.imshow(original)

plt.show()


numpy_image = img_to_array(original)

plt.imshow(np.uint8(numpy_image))

plt.show()
print('numpy array size',numpy_image.shape)

image_batch = np.expand_dims(numpy_image, axis=0)

print('image batch size', image_batch.shape)

plt.imshow(np.uint8(image_batch[0]))
