from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE


def plotImages(images_arr):
    _, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def createmodel(train, validation):
    batch_size = 128

    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5)
    validation_image_generator = ImageDataGenerator(
        rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train,
                                                               shuffle=True,
                                                               target_size=(
                                                                   IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation,
                                                                  target_size=(
                                                                      IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='categorical')
    # _, data = next(val_data_gen)
    # print(data)
    # exit(0)
    total_train = len(next(train_data_gen))
    total_val = len(next(val_data_gen))
    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plotImages(augmented_images)

    # 该模型由三个卷积块组成，每个卷积块中都有一个最大池层。有一个完全连接的层，上面有512个单元，可通过relu激活功能激活该层。该模型根据sigmoid激活函数基于二进制分类输出类概率。
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(val_data_gen.class_indices), activation='softmax')
    ])

    # 对于本教程，请选择ADAM优化器和二进制交叉熵损失函数。要查看每个训练时期的训练和验证准确性，请传递metrics参数。
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # 使用模型的summary方法查看网络的所有层
    model.summary()

    # 使用课程的fit_generator方法ImageDataGenerator来训练网络
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val
    )

    # 保存模型
    model.save("result/mymodel.h5")
    # 保存标签
    createjson("result/mymodel.json", batch_size, epochs, IMG_HEIGHT,
               IMG_WIDTH, val_data_gen.class_indices)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def createjson(jsonname, batch_size, epochs, IMG_HEIGHT, IMG_WIDTH,  class_indices):
    dict = {}
    dict["batch_size"] = batch_size
    dict["epochs"] = epochs
    dict["IMG_HEIGHT"] = IMG_HEIGHT
    dict["IMG_WIDTH"] = IMG_WIDTH
    dict["class_indices"] = class_indices

    with open(jsonname, "w") as fs:
        json.dump(dict, fs)


def readjson(jsonname):
    with open(jsonname, 'r') as f:
        data = json.load(f)
        return data


def dictToArray(dict):
    arr = [0 for i in range(len(dict))]
    for (d, x) in dict.items():
        arr[x] = d
    return arr


def predict():
    dict = readjson("result/mymodel.json")
    IMG_HEIGHT = dict["IMG_HEIGHT"]
    IMG_WIDTH = dict["IMG_WIDTH"]
    batch_size = dict["batch_size"]
    namearray = dictToArray(dict["class_indices"])

    model = keras.models.load_model('mymodel.h5')
    model.summary()

    image = tf.io.read_file("1.jpg")
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range

    x = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float64')
    x[0] = image

    predictions = model.predict(x, batch_size=batch_size)

    print(predictions)

    name = []
    for prediction in predictions:
        index = np.argsort(-prediction)[0]
        name.append(namearray[index])
    print(name)


createmodel("train", "validation")


# createmodel("data", "test")
# predict()
