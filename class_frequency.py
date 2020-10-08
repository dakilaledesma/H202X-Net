from pathlib import Path
import json
import os
import numpy as np
import pickle

image_fp = list(Path("data/nybg2020/train/images/").rglob("*.jpg"))
image_fp = [str(fp) for fp in image_fp]
print(len(image_fp))

train_metadata = open("data/nybg2020/train/metadata.json", 'r').read()
train_metadata = json.loads(train_metadata)

classification = {}
for sample in train_metadata["annotations"]:
    classification[str(sample['id'])] = sample['category_id']

labels = []
for image_fn in image_fp:
    basename = os.path.basename(str(image_fn)).replace(".jpg", "")
    try:
        class_label = classification[basename]
    except KeyError:
        class_label = int(classification[str(int(basename))])
    labels.append(class_label)

# image_fp = np.array(image_fp)
# np.save("data/image_fps.npy", image_fp)

class_freq = {}
for item in labels:
    try:
        class_freq[item] += 1
    except KeyError:
        class_freq[item] = 1

print(class_freq)
cf_file = open('class_frequency', 'wb')
pickle.dump(class_freq, cf_file)
cf_file.close()