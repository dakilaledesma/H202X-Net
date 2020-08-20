from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np

# from runai.ga.tensorflow.keras.optimizers import Optimizer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K
from classification_models.keras import Classifiers

# TF Prep
devices = tf.config.experimental.list_physical_devices('GPU')
print(devices)
for d in devices:
    tf.config.experimental.set_memory_growth(d, True)

# Getting model
SEResNeXt50, preprocess_input = Classifiers.get('seresnext50')

# Config class
class Config:
    model_name = 'seresnext50_ga'
    num_epochs = 1  # for commit, original = 20
    batch_size = 64
    learning_rate = 1e-2
    num_grad_accumulates = 2
    image_size = 32
    step_summary_output = 10

config = Config()

batch_size = 2
image_fp = np.load("data/image_fps.npy")
labels = np.load("data/labels.npy")
labels = to_categorical(labels, dtype=np.bool)

image_fp, labels = shuffle(image_fp, labels)
train_gen = Custom_Generator(image_fp, labels, batch_size)

acc_opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=64)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="cp/seresnext50-1",
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

steps = int(image_fp.shape[0] // batch_size)
model = SEResNeXt50(weights=None, include_top=True, input_shape=(340, 500, 3), classes=32094)
x = keras.layers.Dense(512)(model.output)
output = keras.layers.Dense(32094, activation='softmax')(x)
model = keras.models.Model(inputs=[model.input], outputs=[output])
model.compile(optimizer=acc_opt, loss="categorical_crossentropy")
model.fit_generator(generator=train_gen,
                    steps_per_epoch=int(image_fp.shape[0] // batch_size),
                    epochs=10,
                    verbose=1,
                    callbacks=[model_checkpoint_callback])
model.save("models\\seresnext50-1")