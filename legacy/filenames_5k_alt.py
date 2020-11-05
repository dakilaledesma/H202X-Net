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

labels = []
images = []
for image_fn in image_fp:
    basename = os.path.basename(str(image_fn)).replace(".jpg", "")
    try:
        class_label = classification[basename]
    except KeyError:
        class_label = int(classification[str(int(basename))])

    if int(class_label) < 5000:
        images.append(image_fn)
        labels.append(class_label)

print(len(images))

images = np.array(images)
np.save("data/image_fps_5ka.npy", images)

labels = np.array(labels)
np.save("data/labels_5ka.npy", labels)