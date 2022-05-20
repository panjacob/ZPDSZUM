import os

from utilis.statistics import make_plots, save_model, make_confusion_matrix
from utilis.generator import  get_generators
from utilis.metrics import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf

RESULT_FILENAME = "Sequential_3_128px"  # nazwa pliku gdzie zostana zapisane wyniki w "files/results/{RESULT_FILENAME}"
test_model = "Sequential_3_128px, steps=100, epoch=20, loss=sparse_categorical_crossentropy, optimizer=rmsprop "  # Dodawany do wykresow
LEARN_MODEL_TRUE_OR_LOAD_FALSE = True


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
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
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=metrics
    )
    return model


train_generator, test_generator = get_generators(dataset_name="Data_08", train_size=0.8, image_width=128)
destination_path = os.path.join('files', 'results', RESULT_FILENAME)
model = get_model()
if LEARN_MODEL_TRUE_OR_LOAD_FALSE:
    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # 1000
        epochs=20,  # 100
        validation_data=test_generator,
        validation_steps=100  # 8000
    )

    make_plots(history.history, test_model, destination_path)
    make_confusion_matrix(model, test_generator, destination_path)
    save_model(model, destination_path)
else:

    model_path = os.path.join(destination_path, 'model.h5')
    print(model_path)
    print(os.listdir(destination_path))
    # model = keras.models.load_model(model_path)
    model.load_weights(model_path)
