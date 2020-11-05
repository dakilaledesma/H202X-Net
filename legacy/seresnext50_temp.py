from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from typing import List, Optional, Dict, Generator, NamedTuple, Any, Tuple, Union, Mapping
import itertools

import tensorflow.keras.backend as K
from classification_models.keras import Classifiers
import datetime

# TF Prep
devices = tf.config.experimental.list_physical_devices('GPU')
print(devices)
for d in devices:
    tf.config.experimental.set_memory_growth(d, True)

# Getting model
SEResNeXt50, preprocess_input = Classifiers.get('seresnext50')

# Config class
class Config:
    model_name = 'seresnext50_ga'
    num_epochs = 1  # for commit, original = 20
    batch_size = 4
    learning_rate = 1e-2
    num_grad_accumulates = 64
    image_size = 340
    step_summary_output = 10


# Model class
class Model(tf.keras.Model):
    def __init__(self, num_outputs: int) -> None:
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape) -> None:
        self.core = SEResNeXt50(
            include_top=True,
            weights=None,
            input_shape=(340, 500, 3),
            classes=512,
        )
        self.output_layer = tf.keras.layers.Dense(self.num_outputs)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        y = self.core(x)
        y = self.output_layer(y)
        return y


config = Config()
model = Model(num_outputs=32094)
# build
input_shape = (340, 500, 3)
input = tf.keras.layers.Input(shape=input_shape, name='input_layer', dtype=tf.float32)
_ = model(input)
model.summary()


# Accumulated gradient stuff
def accumulated_gradients(gradients: Optional[List[tf.Tensor]],
                          step_gradients: List[Union[tf.Tensor, tf.IndexedSlices]],
                          num_grad_accumulates: int) -> tf.Tensor:
    if gradients is None:
        gradients = [flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += flat_gradients(g) / num_grad_accumulates

    return gradients


# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices


# Defining training params and funcs
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(config.learning_rate)
train_loss = tf.keras.metrics.Mean('loss/train', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy/train')
metrics = [train_loss, train_accuracy]

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/{config.model_name}/{current_time}'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


def train(config: Config,
          dataset: tf.data.Dataset,
          model: Model):
    global_step = 0
    for e in range(config.num_epochs):
        global_step = train_epoch(config, dataset, model, global_step)
        print(f'{e + 1} epoch finished. step: {global_step}')


def train_epoch(config: Config,
                dataset: tf.data.Dataset,
                model: Model,
                start_step: int = 0) -> tf.Tensor:
    '''Train 1 epoch
    '''
    gradients = None
    global_step = start_step
    for i, batch in enumerate(dataset):
        dummy_step = i + start_step * config.num_grad_accumulates
        x_train, y_train = batch
        print(x_train.shape, y_train.shape)
        step_gradients = train_step(x_train, y_train, loss_fn, optimizer)
        gradients = accumulated_gradients(gradients, step_gradients, config.num_grad_accumulates)
        if (dummy_step + 1) % config.num_grad_accumulates == 0:
            gradient_zip = zip(gradients, model.trainable_variables)
            optimizer.apply_gradients(gradient_zip)
            gradients = None
            if (global_step + 1) % config.step_summary_output == 0:
                write_train_summary(train_summary_writer, metrics, step=global_step + 1)
            global_step += 1

    return global_step


@tf.function
def train_step(x_train: tf.Tensor,
               y_train: tf.Tensor,
               loss_fn: tf.keras.losses.Loss,
               optimizer: tf.keras.optimizers.Optimizer):
    '''Train 1 step and return gradients
    '''
    with tf.GradientTape() as tape:
        outputs = model(x_train, training=True)
        loss = tf.reduce_mean(loss_fn(y_train, outputs))
    train_loss(loss)
    train_accuracy(y_train, tf.nn.softmax(outputs))
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


def write_train_summary(writer: tf.summary.SummaryWriter,
                        metrics: List[tf.keras.metrics.Metric],
                        step: int) -> None:
    with writer.as_default():
        for metric in metrics:
            tf.summary.scalar(metric.name, metric.result(), step=step)
            metric.reset_states()


batch_size = 2
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
labels = to_categorical(labels, dtype=np.bool)


# Generator
# def train_gen():
#     for idx in itertools.count(1):
#         batch_x = image_fp[idx]
#         batch_y = labels[idx]
#
#         print(batch_x, batch_y)
#
#         return_x = []
#         for file_name in batch_x:
#             img = image.load_img(file_name, target_size=(340, 500))
#             x = image.img_to_array(img)
#             x = preprocess_input(x)
#             return_x.append(x)
#         return_x = np.array(return_x)
#
#         return return_x, np.array(batch_y)


image_fp, labels = shuffle(image_fp, labels)
"""
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
"""
gen_dataset = tf.data.Dataset.from_tensor_slices((image_fp, labels))
print(image_fp[0])


def im_file_to_tensor(file, label):
    def _im_file_to_tensor(f, la):
        im = image.load_img(f, target_size=(340, 500))
        im = image.img_to_array(im)
        im = preprocess_input(im)
        return im, la

    file, label = tf.py_function(_im_file_to_tensor,
                                 inp=(file, label),
                                 Tout=(tf.float32, tf.uint8))
    file.set_shape([340, 500, 3])
    label.set_shape([32094])

    print(file.shape, label.shape)
    return file, label

gen_dataset.map(im_file_to_tensor)
train(config, gen_dataset, model)

# batch_size = 2
# image_fp = np.load("data/image_fps.npy")
# labels = np.load("data/labels.npy")
# labels = to_categorical(labels, dtype=np.bool)
#
# image_fp, labels = shuffle(image_fp, labels)
# train_gen = Custom_Generator(image_fp, labels, batch_size)
#
# acc_opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=64)
#
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath="cp/seresnext50-1",
#     save_weights_only=False,
#     monitor='loss',
#     mode='min',
#     save_best_only=True)
#
# steps = int(image_fp.shape[0] // batch_size)
# model = SEResNeXt50(weights=None, include_top=True, input_shape=(340, 500, 3), classes=32094)
# x = keras.layers.Dense(512)(model.output)
# output = keras.layers.Dense(32094, activation='softmax')(x)
# model = keras.models.Model(inputs=[model.input], outputs=[output])
# model.compile(optimizer=acc_opt, loss="categorical_crossentropy")
# model.fit_generator(generator=train_gen,
#                     steps_per_epoch=int(image_fp.shape[0] // batch_size),
#                     epochs=10,
#                     verbose=1,
#                     callbacks=[model_checkpoint_callback])
# model.save("models\\seresnext50-1")