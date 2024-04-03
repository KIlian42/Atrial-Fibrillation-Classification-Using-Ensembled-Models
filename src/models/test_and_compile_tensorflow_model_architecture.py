import numpy as np
from tensorflow_models import *

def main():
    # == Data preparation == 

    # x_train = np.random.rand(12, 9000, 1)
    # input_shape = x_train.shape
    # num_classes = 5
    # model = build_model_CNN_Transformer_12lead(
    #     input_shape=input_shape, num_classes=num_classes
    # )

    # x_train = np.random.rand(20000, 2000)
    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    # input_shape = x_train.shape[1:]
    # num_classes = 5
    # model = build_model_CNN_Transformer_1lead(
    #     input_shape=input_shape, num_classes=num_classes
    # )

    x_train = np.random.rand(20000, 2000)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    input_shape = x_train.shape[1:] # x_train.shape[1:]
    num_classes = 5
    model = build_model_CNN_Transformer_1lead(
        input_shape=input_shape, num_classes=num_classes
    )

    model.summary()

if __name__ == "__main__":
    main() 