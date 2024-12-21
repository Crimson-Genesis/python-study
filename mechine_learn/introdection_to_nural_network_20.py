import numpy as np
import tensorflow as tf
import keras
from keras.api.preprocessing.image import img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions  # type: ignore

# from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions  # type: ignore
# from keras.api.models import Model
# from IPython.display import display

file_1 = "./TF_Keras_Classification_Images/01 Umbrella.jpg"
file_2 = "./TF_Keras_Classification_Images/02 Couple.jpg"
file_3 = "./TF_Keras_Classification_Images/03 Ocean.jpg"


def format_image(img_path: str):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    expanded_array = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded_array)
    return preprocessed


preprocessed_image_1 = format_image(file_1)
preprocessed_image_2 = format_image(file_2)
preprocessed_image_3 = format_image(file_3)


def predect(preprocessed_data, model):
    inception_model = model(
        weights="imagenet",
    )
    inception_model.graph = tf.compat.v1.get_default_graph()
    prediction = inception_model.predict(preprocessed_data)
    decoded_prediction = decode_predictions(prediction)
    return decoded_prediction


inc_predicted_image_1 = predect(preprocessed_image_1, InceptionResNetV2)
inc_predicted_image_2 = predect(preprocessed_image_2, InceptionResNetV2)
inc_predicted_image_3 = predect(preprocessed_image_3, InceptionResNetV2)
print(inc_predicted_image_1)
print(inc_predicted_image_2)
print(inc_predicted_image_3)

# vgg19_predicted_image_1 = predect(preprocessed_image_1, VGG19)
# vgg19_predicted_image_2 = predect(preprocessed_image_2, VGG19)
# vgg19_predicted_image_3 = predect(preprocessed_image_3, VGG19)
# print(vgg19_predicted_image_1)
# print(vgg19_predicted_image_2)
# print(vgg19_predicted_image_3)
