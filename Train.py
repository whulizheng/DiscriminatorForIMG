import Utils


def select_classifier(name, input_shape, batch_size):
    if name == "FCN":
        from Classifiers import Classifier_FCN
        classifier = Classifier_FCN.FCN(input_shape, batch_size)
        return classifier
    elif name == "CIS":
        from Classifiers import Classifier_CIS
        classifier = Classifier_CIS.CIS(input_shape, batch_size)
        return classifier
    elif name == "LSB":
        from Classifiers import Classifier_LSB
        classifier = Classifier_LSB.LSB(input_shape, batch_size)
        return classifier
    else:
        print("WRONG CLASSIFIER")
        exit(-1)


config = Utils.readjson("config.json")
input_shape = config["model"]["input_shape"]
batch_size = config["model"]["batch_size"]
training_path = config["data"]["training_path"]
model_path = config["model"]["model_path"]
model_name = config["model"]["classifier"]
epochs = int(config["model"]["epochs"])

x_train, y_train = Utils.read_dataset(training_path, input_shape, max_num=200)


classifier = select_classifier(model_name, input_shape, batch_size)
# 训练
classifier.train(epochs, x_train, y_train)
# 保存模型
classifier.save_model(model_path)
