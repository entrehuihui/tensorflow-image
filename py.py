from __future__ import absolute_import, division, print_function, unicode_literals
import random
import os
import pathlib
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE


def caption_image(image_path, data_root):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return str(image_rel).split("\\")[-1]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def createDS(file):
    data_root = pathlib.Path(file)
    # for item in data_root.iterdir():
    #     print(item)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())

    label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    image_ds = path_ds.map(load_and_preprocess_image,
                           num_parallel_calls=AUTOTUNE)
    # plt.show()
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    ds = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds, len(all_image_paths), len(label_names)


BATCH_SIZE = 32
ds, allimglen, alllabellen = createDS("data")
print(ds)
# 模型
exit(0)
mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False


def change_range(image, label):
    return 2*image-1, label


keras_ds = ds.map(change_range)
# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
image_batch, label_batch = next(iter(keras_ds))
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(alllabellen, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
model.summary()
steps_per_epoch = tf.math.ceil(allimglen/BATCH_SIZE).numpy()


model.fit(ds, epochs=1, steps_per_epoch=steps_per_epoch)

ds1, l1, l2 = createDS("test")

model.evaluate(ds1, steps=1)
