from glob import glob
import pandas as pd
import codecs, json
import numpy as np
import matplotlib.pyplot as plt


train_path = "data/h2021/train"
with codecs.open(f"{train_path}/metadata.json", 'r', encoding='utf-8', errors='ignore') as f:
    metadata_json = json.load(f)
    
train_df = pd.DataFrame(metadata_json['annotations'], dtype=str)
categories = train_df["category_id"]
categories_dict = {int(t): 0 for t in train_df["category_id"]}

totals = 0
for category in categories:
    categories_dict[int(category)] += 1
    totals += 1

print(totals)
plt.plot(sorted(categories_dict.values()))
plt.show()

plt.plot(sorted(categories_dict.values())[:50000])
plt.show()
