from pathlib import Path
import json
import os
import numpy as np

image_fp = list(Path("data/nybg2020/train/images/").rglob("*.jpg"))
image_fp = [str(fp) for fp in image_fp]
print(len(image_fp))

train_metadata = open("data/nybg2020/train/metadata.json", 'r').read()
train_metadata = json.loads(train_metadata)

classification = {}
num_samples_per_cat = {}
for sample in train_metadata["annotations"]:
    classification[str(sample['id'])] = sample['category_id']
    try:
        num_samples_per_cat[sample['category_id']] += 1
    except KeyError:
        num_samples_per_cat[sample['category_id']] = 0

top_5000 = {k: v for k, v in sorted(num_samples_per_cat.items(), key=lambda item: item[1], reverse=True)}
top_5000 = list(top_5000.keys())[:5000]

labels = []
images = []
for image_fn in image_fp:
    basename = os.path.basename(str(image_fn)).replace(".jpg", "")
    try:
        class_label = classification[basename]
    except KeyError:
        class_label = int(classification[str(int(basename))])

    if class_label in top_5000:
        images.append(image_fn)
        labels.append(class_label)

print(len(images))

images = np.array(images)
np.save("data/image_fps_5k.npy", images)

labels = np.array(labels)
np.save("data/labels_5k.npy", labels)