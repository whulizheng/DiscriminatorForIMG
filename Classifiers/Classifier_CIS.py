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
import random
import math

optimizer = Adam(0.0002, 0.5)


def HPF(image, radius=50, n=1):
    """
    高通滤波函数
    :param image: 输入图像
    :param radius: 半径
    :param n: ButterWorth滤波器阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建ButterWorth高通滤波掩模

    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            try:
                mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + pow(radius / d, 2*n))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(
        image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
    return image_filtering


class CIS():
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

    def preprocess(self, x_train):
        from tqdm import tqdm
        new_x_train = []
        print("预处理...")
        for x in tqdm(x_train):
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
        model.add(AveragePooling2D())
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
        self.discriminator.save(path+'\\best_model_CIS.hdf5')
        print('模型已保存到下面目录')
        print(path + '\\best_model_CIS.hdf5')

    def load_model(self, path):
        self.discriminator = keras.models.load_model(path)
        self.discriminator.summary()
