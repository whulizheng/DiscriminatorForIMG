from PIL import Image
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
optimizer = Adam(0.0002, 0.5)


def get_lsb(img):
    width = img.shape[0]
    height = img.shape[1]
    info = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            if img[i, j] % 2:
                info[i][j] = 1
    info = info.reshape((width, height, 1))
    return info


class LSB():
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2,
                         input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()
        output = Input(shape=self.input_shape)
        validity = model(output)
        return Model(output, validity)

    def preprocess(self, x_train):
        from tqdm import tqdm
        new_x_train = []
        print("预处理...")
        for x in tqdm(x_train):
            new_x_train.append(get_lsb(x))
        return np.array(new_x_train)

    def train(self, epochs, x_train, y_train):
        x_train = np.array(self.preprocess(x_train))
        y_train = np.array(y_train)
        self.discriminator.fit(
            x_train, y_train, epochs=epochs, batch_size=self.batch_size)

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
        print(path + '\\best_model_LSB.hdf5')

    def load_model(self, path):
        self.discriminator = keras.models.load_model(path)
        self.discriminator.summary()
