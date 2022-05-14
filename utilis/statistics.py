import os.path

import matplotlib.pyplot as plt
import numpy as np

from utilis.generator import create_path_if_doesnt_exist
from textwrap import wrap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def make_confusion_matrix(model, test_generator, destination):
    labels = [x for x in test_generator.class_indices]
    y_test = test_generator.classes
    y_pred_probabilites = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred_probabilites, axis=1)
    print(labels)

    cm = confusion_matrix(y_test, y_pred_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    # plt.show()
    plt.savefig(os.path.join(destination, "confusion.png"))


def make_plots(history, description, destination):
    create_path_if_doesnt_exist(destination)
    print(history)
    make_plot(history['loss'], history['val_loss'], 'loss.png', 'loss', 'epoch', f"loss - {description}", destination)
    make_plot(history['accuracy'], history['val_accuracy'], 'accuracy.png', 'accuracy', 'epoch',
              f"accuracy - {description}", destination)
    make_plot(history['f_score'], history['val_f_score'], 'f_score.png', 'f_score', 'epoch', f"f_score - {description}",
              destination)
    make_plot(history['AUC_ROC'], history['val_AUC_ROC'], 'AUC_ROC.png', 'AUC_ROC', 'epoch', f"AUC_ROC - {description}",
              destination)
    make_plot(history['kullback_leibler_divergence'], history['val_kullback_leibler_divergence'],
              'kullback_leibler_divergence.png', 'kullback_leibler_divergence', 'epoch',
              f"kullback_leibler_divergence - {description}",
              destination)


def make_plot(data1, data2, filename, ylabel, xlabel, title, destination):
    plt.figure()
    plt.plot(data1, label='training')
    plt.plot(data2, label='validation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("\n".join(wrap(title, 60)))
    plt.legend()
    plt.savefig(os.path.join(destination, filename))


def save_model(model, destination):
    model.save_weights(os.path.join(destination, 'model.h5'))
