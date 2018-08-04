from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    num_classes = output_shape[0]
    num_convs = 2
    drop_p = 0.5

    model = Sequential()
    # Don't forget to pass input_shape to the first layer of the model
    ##### Your code below (Lab 1)
    
    num_classes = output_shape[0]

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    for i in range(num_convs - 1):
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(drop_p))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(drop_p))
    model.add(Dense(num_classes, activation='softmax'))

    ##### Your code above (Lab 2)

    return model

