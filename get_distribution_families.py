from glob import glob
import pandas as pd
import codecs, json
import numpy as np
from collections import defaultdict


train_path = "data/h2021/train"
with codecs.open(f"{train_path}/metadata.json", 'r', encoding='utf-8', errors='ignore') as f:
    metadata_json = json.load(f)

family_dict = defaultdict(set)
categories_df = pd.DataFrame(metadata_json['categories'], dtype=str)
for family, category in zip(categories_df["family"], categories_df["id"]):
    family_dict[family].add(category)


print(list(family_dict.items())[:5])

train_df = pd.DataFrame(metadata_json['annotations'], dtype=str)
print(train_df.columns)
for family in family_dict.keys():
    categories = family_dict[family]

    num_samples = []
    for category in list(categories):
        num_samples.append(len(train_df.loc[train_df['category_id'] == category]))

    print(f"{family}: Max = {max(num_samples)} | Min = {min(num_samples)} | Diff = {max(num_samples) - min(num_samples)}")