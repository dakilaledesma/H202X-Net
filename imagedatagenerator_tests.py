from keras.preprocessing import image
import numpy as np
import pandas as pd
from matplotlib import pyplot

image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
print(min(labels), max(labels))
labels = np.array(labels, dtype=np.str)
file_df = pd.DataFrame(list(zip(image_fp, labels)), columns=["filename", "class"])
print(file_df.head())

datagen = image.ImageDataGenerator(horizontal_flip=True, zoom_range=[0.85, 0.85])
train_gen = datagen.flow_from_dataframe(file_df, target_size=(500, 340), shuffle=True, class_mode="categorical",
                                        batch_size=1)

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = train_gen.next()
    image = batch[0][0].astype('uint8')
    pyplot.imshow(image)

pyplot.show()
