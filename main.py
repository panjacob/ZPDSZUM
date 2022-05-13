import os

os.environ["KERAS_BACKEND"] = "plaidml.bridge.keras"
import keras
# KERAS_BACKEND=plaidml.keras.backend
from generator import get_generators
from model import get_model

train_generator, test_generator = get_generators("testy", is_create_new_dataset=True, limit_count=True)
model = get_model()
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    validation_data=test_generator,
    validation_steps=800)
model.save_weights('first_try.h5')
