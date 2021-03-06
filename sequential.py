import os

from utilis.statistics import make_plots, save_model, make_confusion_matrix
from utilis.generator import get_generators
from utilis.metrics import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf

RESULT_FILENAME = "Sequential_1"  # nazwa pliku gdzie zostana zapisane wyniki w "files/results/{RESULT_FILENAME}"
test_model = "Sequential, steps=100, epoch=10, loss=binary_crossentropy, optimizer=rmsprop "  # Dodawany do wykresow
LEARN_MODEL_TRUE_OR_LOAD_FALSE = True


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
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
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        # optimizer=tf.keras.optimizers.Ftrl(
        #     l1_regularization_strength=0.001,
        #     learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=0.1, decay_steps=10000, decay_rate=0.9)),
        optimizer='rmsprop',
        metrics=metrics
    )
    return model


train_generator, test_generator = get_generators(dataset_name="Data_08", train_size=0.8)
destination_path = os.path.join('files', 'results', RESULT_FILENAME)
model = get_model()
if LEARN_MODEL_TRUE_OR_LOAD_FALSE:
    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # 1000
        epochs=10,  # 100
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
