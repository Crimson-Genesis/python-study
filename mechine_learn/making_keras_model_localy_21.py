import numpy as np
import tensorflow as tf
import keras
from keras.api.preprocessing.image import img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions  # type: ignore

# from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions  # type: ignore
# from keras.api.models import Model
# from IPython.display import display

print("Hello, World...")
