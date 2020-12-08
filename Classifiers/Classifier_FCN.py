# Preprocessor_GAN
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
import random

optimizer = Adam(0.0002, 0.5)


class FCN():
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(512, kernel_size=3, strides=2,
                         input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()
        output = Input(shape=self.input_shape)
        validity = model(output)
        return Model(output, validity)

    def random_batch(self, x_train, y_train):
        data_size = len(x_train)
        batch_x_train = []
        batch_y_train = []
        if data_size >= self.batch_size:
            randlist = random.sample(
                range(0, data_size), self.batch_size)
            for i in randlist:
                batch_x_train.append(x_train[i])
                batch_y_train.append(y_train[i])
        else:
            for i in range(self.batch_size):
                batch_x_train.append(x_train[i % data_size])
                batch_y_train.append(y_train[i % data_size])
        batch_x_train = np.array(batch_x_train)
        batch_x_train = batch_x_train.reshape((
            self.batch_size,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2]
        ))
        batch_y_train = np.array(batch_y_train)
        return batch_x_train, batch_y_train

    def train(self, epochs, x_train, y_train):
        d_loss = []
        for epoch in range(epochs):
            batch_x_train, batch_y_train = self.random_batch(x_train, y_train)
            loss = self.discriminator.train_on_batch(
                batch_x_train, batch_y_train)
            print("epoch: %d [loss: %f]" % (epoch, loss[0]))
            d_loss.append(loss[0])
        return d_loss

    def discriminate(self, img):
        img = img.reshape([1]+list(img.shape))
        output = self.discriminator(img)
        if output <= 0.5:
            return 0
        else:
            return 1

    def save_model(self, path):
        self.discriminator.save(path+'\\best_model_FCN.hdf5')
        print('模型已保存到下面目录')
        print(path + '\\best_model_FCN.hdf5')

    def load_model(self, path):
        self.discriminator = keras.models.load_model(path)
        self.discriminator.summary()
