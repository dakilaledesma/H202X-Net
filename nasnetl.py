from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
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
image_fp = np.load("data/image_fps_5ka.npy")
labels = np.load("data/labels_5ka.npy")
labels = to_categorical(labels, dtype=np.bool)

image_fp, labels = shuffle(image_fp, labels)
train_gen = Custom_Generator(image_fp, labels, batch_size)

model = NASNetLarge(weights=None, include_top=True, input_shape=(331, 331, 3), classes=5000)
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit_generator(generator=train_gen,
                    steps_per_epoch=int(image_fp.shape[0] // batch_size),
                    epochs=1,
                    verbose=1)
model.save("models\\nasnetl-1")
