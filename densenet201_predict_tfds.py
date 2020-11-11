from tensorflow.keras.models import load_model
from tensorflow import Graph, Session
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer


model = load_model("models\\densenet201-7-tf1")
image_fp = list(Path("data/nybg2020/test/images/").rglob("*.jpg"))

csv_string = "Id,Predicted\n"

imgs = []

image_fp.sort()
# split_imgs = np.array(np.array_split(image_fp, 3200))



def generator():
    i = 0
    while i < len(image_fp) :
        fp = image_fp[i]
        fname = os.path.basename(str(fp)).replace(".jpg", '')
        yield fp, fname
        i += 1

def parse_function(filepath, filename):
    image_string = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = preprocess_input(image)
    image = tf.image.resize(image, [320, 320])
    return image, filename


def train_preprocess(image, filename):
    image = tf.expand_dims(image, axis=0)
    print(image.shape)
    return image, filename


tfds = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string),
                                      output_shapes=(None, None))
tfds = tfds.map(parse_function, num_parallel_calls=20).map(train_preprocess, num_parallel_calls=20)
tfds = tfds.prefetch(10).make_one_shot_iterator()

csv_str_list = []

for _ in tqdm(image_fp):
    tens_, file_name = tfds.get_next()
    fname = os.path.basename(str(file_name)).replace(".jpg", '')
    # proto_tensor = tf.make_ndarray(tens_)
    # print(type(proto_tensor), proto_tensor)
    preds = model.predict(tens_, steps=1)

    csv_str_list.append(f"{fname},{np.argmax(preds)}")

csv_str_list.sort()
csv_preds = "\n".join(csv_str_list)
csv_string += csv_preds

output = open("outputs/densenet201-7-tf1-20.txt", 'w')
output.write(csv_string)
output.close()

# for file_name in tqdm(image_fp):
#     img = image.load_img(file_name, target_size=(320, 320))
#     x = image.img_to_array(img)
#     x = efn.preprocess_input(x)
#     imgs.append(x)

# print(model.predict(imgs))



