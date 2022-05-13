import os

import sys
# AMD GPU
if sys.version.split(' ')[0] == "3.8.6":
    os.environ["KERAS_BACKEND"] = "plaidml.bridge.keras"
    import plaidml.keras

    plaidml.keras.install_backend()

from generator import get_generators
from model import get_model

train_generator, test_generator = get_generators("testy", is_create_new_dataset=True, limit_count=False)
model = get_model()
model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=2,
    validation_data=test_generator,
    validation_steps=800)
model.save_weights('first_try.h5')
