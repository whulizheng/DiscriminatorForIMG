# Preprocessor_GAN
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import cv2
import numpy as np
from scipy import ndimage
import random

optimizer = Adam(0.0002, 0.5)


def HPF(img):
    # 傅里叶变换
    img = img.astype(np.float32)
    res = img
    dft0 = cv2.dft(img[:, :, 0], flags=cv2.DFT_COMPLEX_OUTPUT)
    dft1 = cv2.dft(img[:, :, 1], flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img[:, :, 2], flags=cv2.DFT_COMPLEX_OUTPUT)
    res3 = img[:, :, 3]

    fshift0 = np.fft.fftshift(dft0)
    fshift1 = np.fft.fftshift(dft1)
    fshift2 = np.fft.fftshift(dft2)

    # 设置高通滤波器
    rows = img.shape[0]
    cols = img.shape[1]

    crow, ccol = int(rows/2), int(cols/2)  # 中心位置

    mask = np.ones((rows, cols, 2), np.uint8)

    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # 掩膜图像和频谱图像乘积

    f0 = fshift0 * mask
    f1 = fshift1 * mask
    f2 = fshift2 * mask

    # 傅里叶逆变换

    ishift0 = np.fft.ifftshift(f0)
    ishift1 = np.fft.ifftshift(f1)
    ishift2 = np.fft.ifftshift(f2)

    iimg0 = cv2.idft(ishift0)
    iimg1 = cv2.idft(ishift1)
    iimg2 = cv2.idft(ishift2)

    res0 = cv2.magnitude(iimg0[:, :, 0], iimg0[:, :, 1])
    res1 = cv2.magnitude(iimg1[:, :, 0], iimg1[:, :, 1])
    res2 = cv2.magnitude(iimg2[:, :, 0], iimg2[:, :, 1])

    res[:, :, 0] = res0
    res[:, :, 1] = res1
    res[:, :, 2] = res2
    res[:, :, 3] = res3
    return res


class CIS():
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

    def preprocess(self, x_train):
        new_x_train = []
        for x in x_train:
            new_x_train.append(HPF(x))
        return np.array(new_x_train)

    def build_discriminator(self):
        model = Sequential()
        # feature fusion
        model.add(Conv2D(128, kernel_size=3, strides=2,
                         input_shape=self.input_shape, padding="same"))
        model.add(PReLU())  # PReLU
        # TYpe 1 block
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D())
        # TYpe 1 block
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D())
        # TYpe 2 block
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D())
        # TYpe 2 block
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()
        output = Input(shape=self.input_shape)
        validity = model(output)
        return Model(output, validity)

    def random_batch(self, x_train, y_train):
        x_train = self.preprocess(x_train)
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
        self.discriminator.save(path+'\\best_model_CIS.hdf5')
        print('模型已保存到下面目录')
        print(path + '\\best_model_CIS.hdf5')

    def load_model(self, path):
        self.discriminator = keras.models.load_model(path)
        self.discriminator.summary()
