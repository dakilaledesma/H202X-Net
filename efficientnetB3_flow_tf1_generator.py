#!/usr/bin/python -u

import efficientnet.tfkeras as efn
# from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer
import pandas as pd
import pickle
import os

os.environ['TF_KERAS'] = '1'

import sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

batch_size = 128
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
print(min(labels), max(labels))
labels = np.array(labels)


def generator():
    i = 0
    while i < len(image_fp):
        label = np.zeros(32094)
        label[labels[i]] = 1
        yield image_fp[i], label
        i += 1


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [320, 320])
    return image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


tfds = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.float32),
                                      output_shapes=(None, [32094])).shuffle(len(image_fp))
tfds = tfds.map(parse_function, num_parallel_calls=20).map(train_preprocess, num_parallel_calls=20)
tfds = tfds.repeat().batch(batch_size)
tfds = tfds.prefetch(10)

"""
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
"""

model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="cp/efficientnetb3-7-tf1-{epoch:02d}",
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=False)

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')
with strategy.scope():
    '''
    Load model
    '''
    # model = load_model("cp/efficientnetb3-5-bottleneck-01", custom_objects={'AdamAccumulate': AdamAccumulate}, compile=False)

    '''
    Without bottleneck
    '''
    # model = efn.EfficientNetB3(weights=None, include_top=True, input_shape=(320, 500, 3), classes=32093)
    en_model = efn.EfficientNetB3(weights=None, include_top=False, input_shape=(320, 320, 3), pooling='avg')
    model_output = Dense(32094, activation='softmax')(en_model.output)
    model = Model(inputs=en_model.input, outputs=model_output)

    '''
    With bottleneck
    '''
    # en_model = efn.EfficientNetB3(weights='noisy-student', include_top=False, input_shape=(320, 320, 3), pooling='avg')
    # model_output = Dense(512, activation='linear')(en_model.output)
    # model_output = Dense(32094, activation='softmax')(model_output)
    # model = Model(inputs=en_model.input, outputs=model_output)

    # model = Model(inputs=en_model.input, outputs=model_output)
    model.compile(optimizer='adam', loss="categorical_crossentropy")



model.summary()
model.fit(tfds,
          steps_per_epoch=int(image_fp.shape[0] // batch_size),
          epochs=12,
          verbose=1,
          callbacks=[model_checkpoint_callback], max_queue_size=100, workers=20, use_multiprocessing=True)

model.save("models\\efficientnetb3-7-tf1")
