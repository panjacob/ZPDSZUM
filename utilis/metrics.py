from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import metrics
from keras import backend as K


def precision_(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall_(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


metrics = [
    'accuracy',
    metrics.AUC(name='AUC_ROC', curve="ROC"),
    f_score,
    #     Dodatkowe 5
    'kullback_leibler_divergence',
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.losses.CategoricalCrossentropy(),
    tf.keras.losses.Poisson(),
]
