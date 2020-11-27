import Utils


def select_classifier(name, input_shape, batch_size):
    if name == "FCN":
        import Classifier_FCN as FCN
        classifier = FCN.FCN(input_shape, batch_size)
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

x_train, y_train = Utils.read_dataset(training_path, input_shape)
classifier = select_classifier(model_name, input_shape, batch_size)
# 训练
d_loss = classifier.train(epochs, x_train, y_train)
# 保存模型
classifier.save_model(model_path)
