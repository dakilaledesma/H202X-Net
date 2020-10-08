import efficientnet.keras as efn
from keras.preprocessing import image
# from tensorflow.keras.applications.nasnet import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from keras_gradient_accumulation import GradientAccumulation

# from runai.ga.keras.optimizers import Optimizer
# from keras.models import load_model
# from keras.optimizers import Adam

import keras
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from ml_libs.cosine_annealing import CosineAnnealingScheduler
import pandas as pd
import pickle

batch_size = 8
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
print(min(labels), max(labels))
labels = np.array(labels, dtype=np.str)
# labels = to_categorical(labels, dtype=np.int)
# print(len(labels[0]))
# image_fp, labels = shuffle(image_fp, labels)

file_df = pd.DataFrame(list(zip(image_fp, labels)), columns=["filename", "class"])
print(file_df.head())

datagen = image.ImageDataGenerator(horizontal_flip=True, zoom_range=[0.85, 0.85], preprocessing_function=efn.preprocess_input)
train_gen = datagen.flow_from_dataframe(file_df, target_size=(320, 320), shuffle=True, class_mode="categorical", batch_size=batch_size)
pickled_classes = open('eb3_traingen_classes', 'wb')
pickle.dump(train_gen.class_indices, pickled_classes)
pickled_classes.close()
# train_gen = Custom_Generator(image_fp, labels, batch_size)
# print(train_gen.class_indices)

optimizer = GradientAccumulation('sgd', accumulation_steps=128)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="cp/efficientnetb3-6-bottleneck-{epoch:02d}",
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

'''
Load model
'''
# model = load_model("cp/efficientnetb3-5-bottleneck-01", custom_objects={'AdamAccumulate': AdamAccumulate}, compile=False)

'''
Without bottleneck
'''
# model = efn.EfficientNetB3(weights=None, include_top=True, input_shape=(320, 320, 3), classes=32093)
# en_model = efn.EfficientNetB3(weights='noisy-student', include_top=False, input_shape=(320, 320, 3), pooling='avg')
# model_output = Dense(32093, activation='softmax')(en_model.output)
# model = Model(inputs=en_model.input, outputs=model_output)

'''
With bottleneck
'''
en_model = efn.EfficientNetB3(weights='noisy-student', include_top=False, input_shape=(320, 320, 3), pooling='avg')
model_output = Dense(512, activation='linear')(en_model.output)
model_output = Dense(32093, activation='softmax')(model_output)
model = Model(inputs=en_model.input, outputs=model_output)


# model = Model(inputs=en_model.input, outputs=model_output)
model.compile(optimizer=acc_opt, loss="categorical_crossentropy")
model.summary()

model.fit_generator(generator=train_gen,
                    steps_per_epoch=int(image_fp.shape[0] // batch_size),
                    epochs=12,
                    verbose=1,
                    callbacks=[model_checkpoint_callback,
                               CosineAnnealingScheduler(T_max=2, eta_max=1e-2, eta_min=1e-4)])

model.save("models\\efficientnetb3-6-bottleneck")