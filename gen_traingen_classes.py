from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
import numpy as np
import pandas as pd
import pickle

batch_size = 6
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
print(min(labels), max(labels))
labels = np.array(labels, dtype=np.str)
# labels = to_categorical(labels, dtype=np.int)
# print(len(labels[0]))
# image_fp, labels = shuffle(image_fp, labels)

file_df = pd.DataFrame(list(zip(image_fp, labels)), columns=["filename", "class"])
print(file_df.head())

datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_dataframe(file_df, target_size=(340, 500), shuffle=True, class_mode="categorical", batch_size=batch_size)
print(train_gen.class_indices)

pickled_classes = open('traingen_classes', 'ab')
pickle.dump(train_gen.class_indices, pickled_classes)
pickled_classes.close()
