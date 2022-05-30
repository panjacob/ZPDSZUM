import os

from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras import Model

from utilis.generator_cat_dogs import get_generators
from utilis.metrics import metrics
from utilis.statistics import make_plots, make_confusion_matrix, save_model, get_wrong_classified

RESULT_FILENAME = "dogs_cats2"  # nazwa pliku gdzie zostana zapisane wyniki w "files/results/{RESULT_FILENAME}"
test_model = "MobilNet_2_224px_long, steps=?, epoch=20, loss=sparse_categorical_crossentropy, optimizer=adamax=0.001"  # Dodawany do wykresow
LEARN_MODEL_TRUE_OR_LOAD_FALSE = True


# initial weights to the imagenet weights
# adjustable learning rate. The keras callback ReduceLROnPlateau
# https://stackoverflow.com/questions/63347272/good-training-validation-accuracy-but-poor-test-accuracy
def get_model():
    lr = 0.001
    mobile = tf.keras.applications.mobilenet.MobileNet(include_top=False,
                                                       input_shape=(64, 64, 3),
                                                       pooling='max', weights='imagenet',
                                                       alpha=1, depth_multiplier=1,
                                                       # dropout=.5
                                                       )
    x = mobile.layers[-1].output
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    model.compile(tf.keras.optimizers.Adamax(lr=lr), loss='categorical_crossentropy', metrics=metrics)

    lr_adjust = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=0, mode="auto",
                                                     min_delta=0.00001, cooldown=0, min_lr=0)
    callbacks = [lr_adjust]
    return model, callbacks


train_generator, test_generator = get_generators(dataset_name="cat_dog", image_width=64, )
destination_path = os.path.join('files', 'results', RESULT_FILENAME)
model, callbacks = get_model()
if LEARN_MODEL_TRUE_OR_LOAD_FALSE:
    history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=callbacks)
    print(history)

    make_plots(history.history, test_model, destination_path)
    make_confusion_matrix(model, test_generator, destination_path)
    save_model(model, destination_path)
else:
    model_path = os.path.join(destination_path, 'model.h5')
    model.load_weights(model_path)
    get_wrong_classified(model, test_generator)
