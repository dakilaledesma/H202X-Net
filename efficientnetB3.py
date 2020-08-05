import efficientnet.tfkeras
import keras
from runai.ga.keras.optimizers import Optimizer
from keras.models import load_model
from keras.optimizers import Adam

import efficientnet.keras as efn
from keras.preprocessing import image
# from tensorflow.keras.applications.nasnet import preprocess_input
from keras.utils import to_categorical


from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np


class Custom_Generator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return_x = []
        for file_name in batch_x:
            img = image.load_img(file_name, target_size=(320, 320))
            x = image.img_to_array(img)
            x = efn.preprocess_input(x)
            return_x.append(x)
        return_x = np.array(return_x)

        return return_x, np.array(batch_y)


batch_size = 64
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
labels = to_categorical(labels, dtype=np.bool)

image_fp, labels = shuffle(image_fp, labels)
train_gen = Custom_Generator(image_fp, labels, batch_size)

steps = int(image_fp.shape[0] // batch_size)
model = efn.EfficientNetB3(weights=None, include_top=True, input_shape=(320, 320, 3), classes=32094)
model.compile(optimizer=Optimizer(Adam(), steps=steps), loss="categorical_crossentropy")
model.fit_generator(generator=train_gen,
                    steps_per_epoch=steps,
                    epochs=75,
                    verbose=1)
model.save(f"models\\efficientnetb3-2-full")
