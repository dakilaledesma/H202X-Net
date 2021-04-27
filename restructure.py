import json
import os
import shutil
import pandas as pd
import codecs
from glob import glob
from tqdm import tqdm

restructure_path = "E:/h2021/restructured/train/images"

train_path = "E:/h2021/train"
with codecs.open(f"{train_path}/metadata.json", 'r', encoding='utf-8', errors='ignore') as f:
    metadata_json = json.load(f)

train_df = pd.DataFrame(metadata_json['annotations'], dtype=str)
train_df = train_df[["category_id", "image_id"]]
image_categories = dict(zip(train_df["image_id"], train_df["category_id"]))

categories = set(image_categories.values())
for category in tqdm(categories, desc="Creating folders"):
    os.makedirs(f"{restructure_path}/{category}")

image_filepaths = glob(f"{train_path}/**/*.jpg", recursive=True)
for filepath in tqdm(image_filepaths, desc="Moving images"):
    image_id = os.path.basename(filepath).replace(".jpg", '')
    image_category = image_categories[str(int(image_id))]

    shutil.move(filepath, f"{restructure_path}/{image_category}/{image_id}.jpg")

verify = glob(f"{restructure_path}/**/*.jpg", recursive=True)
print(len(verify))