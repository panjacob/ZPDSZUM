import os

from utilis.statistics import make_plots, save_model
from utilis.generator import get_generators
from utilis.metrics import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

RESULT_FILENAME = "Sequential_1"  # nazwa pliku gdzie zostana zapisane wyniki w "files/results/{RESULT_FILENAME}"
test_model = "Sequential, steps=100, epoch=10, loss=binary_crossentropy, optimizer=rmsprop"  # Dodawany do wykresow


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
        optimizer='rmsprop',
        metrics=metrics
    )
    return model


train_generator, test_generator = get_generators(dataset_name="Data_08", train_size=0.8)
model = get_model()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 1000
    epochs=10,  # 100
    validation_data=test_generator,
    validation_steps=100  # 8000
)
destination_path = os.path.join('files', 'results', RESULT_FILENAME)
make_plots(history.history, test_model, destination_path)
save_model(model, destination_path)
