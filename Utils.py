import json


def readjson(address):
    with open(address, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def read_dataset(path, input_shape, max_num=None):
    import cv2
    import glob
    x_train = []
    y_train = []
    # 文件名称应该规范为 xx_x.png，前面的xx为序号，后面的x为标签
    if max_num is None:
        for img_file in glob.glob(path+r'/*.png'):
            try:
                tag = img_file[-5]
                src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                if src.shape != input_shape:
                    src = convert_img(
                        src, input_shape[0], input_shape[1], input_shape[2])

                ###
                x_train.append(src)
                y_train.append(int(tag))
                ###
            except Exception as e:
                print(e)
        return x_train, y_train
    else:
        count = 0
        for img_file in glob.glob(path+r'/*.png'):
            try:
                tag = img_file[-5]
                src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                if src.shape != tuple(input_shape):
                    src = convert_img(
                        src, input_shape[0], input_shape[1])

                ###
                x_train.append(src)
                y_train.append(int(tag))
                count += 1
                ###
            except Exception as e:
                print(e)
            if count == max_num:
                return x_train, y_train
        return x_train, y_train


def convert_img(src, width=128, height=128, chanel=4):
    import cv2
    try:
        src = cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)
        return src
    except Exception as e:
        print(e)
        exit()
