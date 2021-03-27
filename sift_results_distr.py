from glob import glob
import pandas as pd
import codecs, json
import random
import numpy as np

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
# print(sum(list(categories_dict.values())))
categories_val = list(categories_dict.values())
# print(min(categories_dict.values()), max(categories_dict.values()))
num_scale = np.interp(categories_val, (min(categories_val), max(categories_val)), (1, 10))

categories_dict = {i: round(num_scale[i]) for i in range(64500)}
print(sum(list(categories_dict.values())), min(categories_dict.values()))



results = open("submission_files/topk_ids.csv")
result_lines = results.readlines()

best_outlines = []
best_numclasses = 0
patience_counter = 0

rand_seed = 0
while True:
    out_lines = ["Id,Predicted"]
    out_dict = {i: 0 for i in range(64500)}

    # random.seed(7231970)
    # random.seed(12251997)

    random.seed(rand_seed)
    indices = random.sample(range(0, len(result_lines)), len(result_lines))
    for index in indices:
        line = result_lines[index].replace(".jpg", '')
        vals = line.split(",")

        top_k = [int(x) for x in vals[1:]]

        for i in top_k:
            if out_dict[i] < categories_dict[i]:
                out_dict[i] += 1
                out_lines.append(f"{vals[0]},{i}")
                break
        else:
            out_lines.append(f"{vals[0]},{top_k[0]}")

    numclasses = len(set([key for key, value in out_dict.items() if value > 0]))
    if numclasses > best_numclasses:
        best_numclasses = numclasses
        best_outlines = out_lines
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Run {rand_seed}: {numclasses} | Best: {best_numclasses} | Patience: {patience_counter}")
    if patience_counter >= 1000:
        break

    rand_seed += 1
    break


print(best_numclasses)
out_lines = '\n'.join(best_outlines)
out_file = open("submission_files/out11-l-icf-distr.csv", 'w')
out_file.write(out_lines)
out_file.close()
