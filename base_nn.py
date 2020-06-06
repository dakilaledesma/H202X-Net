import numpy as np
from PIL import Image
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np


class Custom_Generator(tf.keras.utils.Sequence):
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
            img = image.load_img(file_name, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            return_x.append(x)
        return_x = np.array(return_x)

        return return_x, np.array(batch_y)


batch_size = 8
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
labels = to_categorical(labels, dtype='uint8')
train_gen = Custom_Generator(image_fp, labels, batch_size)

model = NASNetLarge(weights=None, include_top=True, input_shape=(331, 331, 3), classes=32094)
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit_generator(generator=train_gen,
                    steps_per_epoch=int(3800 // batch_size),
                    epochs=10,
                    verbose=1)
