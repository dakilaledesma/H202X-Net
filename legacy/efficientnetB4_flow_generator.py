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

class AdamAccumulate(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
    image = tf.image.resize(image, [380, 380])
    return image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


tfds = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.float32),
                                      output_shapes=(None, [32094])).shuffle(len(image_fp))
tfds = tfds.map(parse_function, num_parallel_calls=20).map(train_preprocess, num_parallel_calls=20)
tfds = tfds.batch(batch_size)
tfds = tfds.prefetch(10)

"""
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
"""

model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="cp/EfficientNetB4-7-{epoch:02d}",
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=False)

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')
acc_opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=2)
with strategy.scope():
    '''
    Load model
    '''
    # model = load_model("cp/EfficientNetB4-5-bottleneck-01", custom_objects={'AdamAccumulate': AdamAccumulate}, compile=False)

    '''
    Without bottleneck
    '''
    # model = efn.EfficientNetB4(weights=None, include_top=True, input_shape=(320, 500, 3), classes=32093)
    en_model = efn.EfficientNetB4(weights='noisy-student', include_top=False, input_shape=(380, 380, 3), pooling='avg')
    model_output = Dense(32094, activation='softmax')(en_model.output)
    model = Model(inputs=en_model.input, outputs=model_output)

    '''
    With bottleneck
    '''
    # en_model = efn.EfficientNetB4(weights='noisy-student', include_top=False, input_shape=(380, 380, 3), pooling='avg')
    # model_output = Dense(512, activation='linear')(en_model.output)
    # model_output = Dense(32094, activation='softmax')(model_output)
    # model = Model(inputs=en_model.input, outputs=model_output)

    # model = Model(inputs=en_model.input, outputs=model_output)

    model.compile(optimizer=acc_opt, loss="categorical_crossentropy")

model.summary()
model.fit(tfds,
          steps_per_epoch=int(image_fp.shape[0] // batch_size),
          epochs=12,
          verbose=1,
          callbacks=[model_checkpoint_callback], max_queue_size=100, workers=20, use_multiprocessing=True)

model.save("models\\efficientnetb4-7")
