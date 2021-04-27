import random
from tqdm import tqdm
from collections import defaultdict
import codecs
import json
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm

train_path = "data/h2021/train"
with codecs.open(f"{train_path}/metadata.json", 'r', encoding='utf-8', errors='ignore') as f:
    metadata_json = json.load(f)

train_df = pd.DataFrame(metadata_json['annotations'], dtype=str)
categories = train_df["category_id"]
categories_dict = {int(t): 0 for t in train_df["category_id"]}
totals = 0
for category in categories:
    if categories_dict[int(category)] != 10:
        categories_dict[int(category)] += 1
    totals += 1

num_images = len(glob("data/test/images/*.jpg"))
print(num_images, totals)
coeff = 0.5

categories_dict = dict(sorted(categories_dict.items()))
categories_val = list(categories_dict.values())
num_scale = np.interp(categories_val, (min(categories_val), max(categories_val)), (1, 10))

categories_dict = {i: round(num_scale[i]) for i in range(64500)}
print(sum(list(categories_dict.values())), min(categories_dict.values()))

results = open("submission_files/topk_ids.csv")
result_lines = results.readlines()

ids = {k: 0 for k in range(64500)}
im_to_id = {}
final_im_to_id = {}
for index, line in enumerate(result_lines):
    line = line.replace(".jpg", '')
    vals = line.split(",")


    line = [int(x) for x in vals]
    im, topk = line[0], line[1:]
    im_to_id[im] = topk
    final_im_to_id[im] = topk[0]
    ids[topk[0]] += 1

ims = list(im_to_id.keys())
topks = list(im_to_id.values())
topks = np.array(topks)
print(topks.shape)
for i, val in tqdm(enumerate(list(ids.values()))):
    if val == 0:
        for j in range(5):
            arr = topks[:, j]
            columnk_results = np.where(arr == i)[0]
            if len(columnk_results) == 1:
                for columnk in columnk_results:
                    if ids[final_im_to_id[columnk]] > 1:
                        final_im_to_id[columnk] = i

                break

print(len(set(list(final_im_to_id.values()))))

out_lines = ["Id,Predicted"]
for k, v in final_im_to_id.items():
    out_lines.append(f"{k},{v}")

out_lines = '\n'.join(out_lines)
out_file = open("submission_files/out23_srxaug30_b.csv", 'w')
out_file.write(out_lines)
out_file.close()
