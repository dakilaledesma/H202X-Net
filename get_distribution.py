from glob import glob
import pandas as pd
import codecs, json
import numpy as np


train_path = "data/h2021/train"
with codecs.open(f"{train_path}/metadata.json", 'r', encoding='utf-8', errors='ignore') as f:
    metadata_json = json.load(f)
    
train_df = pd.DataFrame(metadata_json['annotations'], dtype=str)
categories = train_df["category_id"]
categories_dict = {int(t): 0 for t in train_df["category_id"]}

num_images = len(glob("data/test"))
print(len(num_images))

totals = 0
for category in categories:
    categories_dict[int(category)] += 1
    totals += 1

categories_dict = dict(sorted(categories_dict.items()))
print(list(categories_dict.items())[:5])
inv = [abs(1 - v/totals) for v in list(categories_dict.values())]
print(inv[:5])

inv = np.array(inv)
np.save('inverse_cf.npy', inv)