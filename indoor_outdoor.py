import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

MODEL_PATH = './models/indoor_outdoor_sgd_01.h5'


def main():
    img_file = sys.argv[1]

    if not os.path.exists(img_file):
        print('file does not exist')

    model = tf.keras.models.load_model(MODEL_PATH)

    img = load_image(img_file)

    prediction = model.predict(img)

    if prediction > 0.5:
        print('outdoors')
    else:
        print('indoors')


def load_image(image_path):
    '''
    input: path to image file 
    output: scaled and resized numpy array

    Need to read in the image and then convert to numpy array
    Then have to add an empty dimension to match model 
    Lastly rescale like was done in training
    '''

    pil_img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(352, 240))

    img = tf.keras.preprocessing.image.img_to_array(pil_img)

    img = img.reshape((1,)+img.shape)
    img = img.astype('float')/255.

    return img


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
