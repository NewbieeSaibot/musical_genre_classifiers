import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "./data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert list
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def plot_history(history):
    fig, axs = plt.subplots(2)
    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    #create error subplot
    #create accuracy subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    #load data
    inputs, targets = load_data(DATA_PATH)
    #split data in train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)
    #building model architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        # 1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.Dropout(0.3),
        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.L2(0.001)),
        keras.layers.Dropout(0.3),
        #output layer
        keras.layers.Dense(10, activation="softmax"),
    ])
    #compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    #train network
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=500, batch_size=32)
    #save the model
    model.save("mlp_model_2")
    plot_history(history)
