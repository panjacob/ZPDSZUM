import os

# from keras_applications.resnet_common import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.layers import GlobalAveragePooling2D

from utilis.statistics import make_plots, save_model, make_confusion_matrix
from utilis.generator import get_generators, get_generators_augumented
from utilis.metrics import metrics

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from tensorflow.keras import Model

RESULT_FILENAME = "ResNet50V2_224px_1_64px_long"  # nazwa pliku gdzie zostana zapisane wyniki w "files/results/{RESULT_FILENAME}"
test_model = "ResNet50V2_1_64px, steps=299, epoch=30, loss=sparse_categorical_crossentropy, optimizer=adam"  # Dodawany do wykresow
LEARN_MODEL_TRUE_OR_LOAD_FALSE = True


def get_model():
    base_model = ResNet50V2(include_top=False,
                            input_shape=(64, 64, 3),
                            weights='imagenet')
    base_model.trainable = True
    add_to_base = base_model.output
    add_to_base = GlobalAveragePooling2D(data_format='channels_last', name='head_gap')(add_to_base)
    new_output = Dense(4, activation='softmax', name='head_pred')(add_to_base)
    model = Model(base_model.input, new_output)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        # optimizer='rmsprop',
        metrics=metrics
    )
    return model


train_generator, test_generator = get_generators_augumented(dataset_name="Data_08_224", train_size=0.8, image_width=64)
destination_path = os.path.join('files', 'results', RESULT_FILENAME)
model = get_model()
if LEARN_MODEL_TRUE_OR_LOAD_FALSE:
    history = model.fit(train_generator, epochs=30, validation_data=test_generator, )
    print(history)
    make_plots(history.history, test_model, destination_path)
    make_confusion_matrix(model, test_generator, destination_path)
    save_model(model, destination_path)
else:
    model_path = os.path.join(destination_path, 'model.h5')
    model.load_weights(model_path)

