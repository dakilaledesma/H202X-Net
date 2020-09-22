# from tensorflow.keras.applications.densenet import DenseNet201
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.nasnet import preprocess_input
# from tensorflow.keras.utils import to_categorical
# from sklearn.utils import shuffle
# import tensorflow as tf
# import numpy as np

# Reading data
import numpy as np
import pandas as pd
import json, codecs
from pathlib import Path

import os
for dirname, _, filenames in os.walk('/data/nybg2020'):
    for filename in filenames:
        if filename.endswith('.jpg'):
            break
        print(os.path.join(dirname, filename))

sample_sub = pd.read_csv('data/sample_submission.csv')
# print(sample_sub)

with codecs.open("data/nybg2020/train/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    train_metadata = json.load(f)

with codecs.open("data/nybg2020/test/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)

print(train_metadata.keys())

image_labels = {}

train_df = pd.DataFrame(train_metadata['annotations'])
print(train_df)

# Getting the names for each family, genus and category_id
train_cat = pd.DataFrame(train_metadata['categories'])
train_cat.columns = ['family', 'genus', 'category_id', 'category_name']
print(train_cat)

train_img = pd.DataFrame(train_metadata['images'])
train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']
print(train_img)

image_fp = list(Path("data/nybg2020/train/images/").rglob("*.jpg"))
image_fp = [str(fp) for fp in image_fp]
print(len(image_fp))

classification = {}
family_ids = {}
genus_ids = {}
family_counter = 0
genus_counter = 0

for sample in train_metadata["annotations"]:
    classification[str(sample['id'])] = sample['category_id']

for image_fn in image_fp:
    basename = os.path.basename(str(image_fn)).replace(".jpg", "")
    try:
        class_label = classification[basename]
    except KeyError:
        class_label = int(classification[str(int(basename))])
    # print(train_cat.loc[train_cat["category_id"] == class_label])

    family = str(train_cat.loc[train_cat["category_id"] == class_label]["family"].to_numpy()[0])
    try:
        family_id = family_ids[family]
    except KeyError:
        family_ids[family] = family_counter
        family_counter += 1
    finally:
        family_id = family_ids[family]

    genus = str(train_cat.loc[train_cat["category_id"] == class_label]["genus"].to_numpy()[0])
    try:
        genus_id = genus_ids[genus]
    except KeyError:
        genus_ids[genus] = genus_counter
        genus_counter += 1
    finally:
        genus_id = genus_ids[genus]

    print(family_id, genus_id, class_label)

    # image_labels[class_label] = list(train_metadata.loc[train_metadata["image_id"] == class_label])
