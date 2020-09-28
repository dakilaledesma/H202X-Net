import keras
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from classification_models import Classifiers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from ml_libs.cosine_annealing import CosineAnnealingScheduler
import pandas as pd
from keras.preprocessing import image

from keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np

# from runai.ga.keras.optimizers import Optimizer
# from keras.models import load_model
# from keras.optimizers import Adam

SEResNeXt50, preprocess_input = Classifiers.get('seresnext50')

class AdamAccumulate(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
train_gen = datagen.flow_from_dataframe(file_df, target_size=(224, 327), shuffle=True, class_mode="categorical", batch_size=batch_size)
# train_gen = Custom_Generator(image_fp, labels, batch_size)
# print(train_gen.class_indices)

"""
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
"""
acc_opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=64)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="cp/seresnext-4",
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

steps = int(image_fp.shape[0] // batch_size)

'''
Without bottleneck
'''
# model = SEResNeXt50(weights=None, input_shape=(224, 327, 3), classes=32093)

'''
With bottleneck
'''
sr_model = SEResNeXt50(weights=None, input_shape=(224, 327, 3), include_top=False)
bottleneck = GlobalAveragePooling2D(name='avg_pool')(sr_model.output)
bottleneck = Dense(512, activation='relu')(bottleneck)
model_output = Dense(32093, activation='softmax', name='fc1000')(bottleneck)

model = Model(inputs=sr_model.input, outputs=model_output)
model.compile(optimizer=acc_opt, loss="categorical_crossentropy")
model.summary()
model.fit_generator(generator=train_gen,
                    steps_per_epoch=int(image_fp.shape[0] // batch_size),
                    epochs=1,
                    verbose=1,
                    callbacks=[model_checkpoint_callback,
                               CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)])

model.save("models\\seresnext-4-bottleneck")