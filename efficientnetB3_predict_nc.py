import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer

from classification_models import Classifiers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from ml_libs.cosine_annealing import CosineAnnealingScheduler
import pandas as pd
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.utils import shuffle
import pickle

# model = load_model("models/efficientnetb3-7-squished-bottleneck-10", custom_objects={'AdamAccumulate': AdamAccumulate}, compile=False)
model = load_model("models/densenet201-7-tf1", compile=False)
# acc_opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=64)
model.compile(loss='categorical_crossentropy', optimizer='adam')
image_fp = list(Path("data/nybg2020/test/images/").rglob("*.jpg"))

csv_string = "Id,Predicted\n"

imgs = []

image_fp.sort()
split_imgs = np.array(np.array_split(image_fp, 3200))

csv_str_list = []


for split in tqdm(split_imgs):
    imgs = []
    fnames = []

    for file_name in split:
        img = image.load_img(file_name, target_size=(320, 320))
        x = image.img_to_array(img)
        # x = preprocess_input(x)
        x = efn.preprocess_input(x)
        imgs.append(x)
        fnames.append(os.path.basename(str(file_name)).replace(".jpg", ''))

    imgs = np.array(imgs)
    preds = model.predict(imgs)

    for a, b in zip(fnames, preds):
        csv_str_list.append(f"{a},{np.argmax(b)}")

csv_str_list.sort()
csv_preds = "\n".join(csv_str_list)
csv_string += csv_preds

output = open("outputs/densenet201-7-tf1-20.txt", 'w')
output.write(csv_string)
output.close()

# for file_name in tqdm(image_fp):
#     img = image.load_img(file_name, target_size=(340, 500))
#     x = image.img_to_array(img)
#     x = efn.preprocess_input(x)
#     imgs.append(x)

# print(model.predict(imgs))



