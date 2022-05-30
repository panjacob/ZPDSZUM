import os

from utilis.statistics import make_plots, save_model, make_confusion_matrix
from utilis.generator_cat_dogs import get_generators
from utilis.metrics import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf

RESULT_FILENAME = "dog_cat_swquential_long"  # nazwa pliku gdzie zostana zapisane wyniki w "files/results/{RESULT_FILENAME}"
test_model = "Sequential_dog_cat_1_64px, steps=?, epoch=20, loss=sparse_categorical_crossentropy, optimizer=adam"  # Dodawany do wykresow
LEARN_MODEL_TRUE_OR_LOAD_FALSE = True


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        # optimizer='rmsprop',
        metrics=metrics
    )
    return model


train_generator, test_generator = get_generators(dataset_name="cat_dog", image_width=64)
destination_path = os.path.join('files', 'results', RESULT_FILENAME)
model = get_model()
if LEARN_MODEL_TRUE_OR_LOAD_FALSE:
    STEPS_PER_EPOCH = 100
    history = model.fit(
        train_generator,
        epochs=20,  # 100
        validation_data=test_generator,
    )

    make_plots(history.history, test_model, destination_path)
    make_confusion_matrix(model, test_generator, destination_path)
    save_model(model, destination_path)
else:

    model_path = os.path.join(destination_path, 'model.h5')
    model.load_weights(model_path)
