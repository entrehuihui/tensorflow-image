from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import json
from tensorflow import keras
import os
import numpy as np
import pyautogui
AUTOTUNE = tf.data.experimental.AUTOTUNE


def readjson(jsonname):
    with open(jsonname, 'r') as f:
        data = json.load(f)
        return data


def dictToArray(dict):
    arr = [0 for i in range(len(dict))]
    for (d, x) in dict.items():
        arr[x] = d
    return arr


def reading(imagesfile, IMG_HEIGHT, IMG_WIDTH):
    image = tf.io.read_file(imagesfile)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range

    x = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float64')
    x[0] = image
    return x


def predict(imagesfiles):
    # 识别
    dict = readjson("result/mymodel.json")
    IMG_HEIGHT = dict["IMG_HEIGHT"]
    IMG_WIDTH = dict["IMG_WIDTH"]
    batch_size = dict["batch_size"]
    namearray = dictToArray(dict["class_indices"])

    model = keras.models.load_model('result/mymodel.h5')
    model.summary()
    result = []
    for imagesfile in imagesfiles:
        x = reading(imagesfile, IMG_HEIGHT, IMG_WIDTH)
        predictions = model.predict(x, batch_size=batch_size)
        # print("%f" % predictions[0][0])
        for prediction in predictions:
            index = np.argsort(-prediction)[0]
            print(namearray[index])
            result.append(namearray[index])

    return result


def screenshot():
    # 获取屏幕截图
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    image = pyautogui.screenshot()
    image = np.array(image)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range
    x = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float64')
    x[0] = image
    return x


result = predict(["2.jpg", "1.jpg"])
# print(result)

screenshot()
