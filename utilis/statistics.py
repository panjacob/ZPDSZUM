import os.path

import matplotlib.pyplot as plt
import numpy as np

from utilis.generator import create_path_if_doesnt_exist
from textwrap import wrap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_wrong_classified(model, generator):
    labels = [x for x in generator.class_indices]

    y_test = generator.classes
    y_pred_probabilites = model.predict(generator)
    y_pred_classes = np.argmax(y_pred_probabilites, axis=1)
    print(labels)

    x = 0
    for i in range(len(y_pred_classes)):
        if y_test[i] != y_pred_classes[i]:
            x += 1
            # print(f"{i + 1}" + ".jpg")
    print("zle", x, "/", len(y_pred_classes))


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
    # W starszej wersji tensorflow jest acc
    try:
        make_plot(history, 'accuracy', description, destination)
    except:
        make_plot(history, 'acc', description, destination)
    make_plot(history, 'loss', description, destination)

    make_plot(history, 'f_score', description, destination)
    make_plot(history, 'AUC_ROC', description, destination)
    # Dodatkowe
    make_plot(history, 'kullback_leibler_divergence', description, destination)
    make_plot(history, 'precision', description, destination)
    make_plot(history, 'recall', description, destination)
    make_plot(history, 'categorical_crossentropy', description, destination, ylim=None)
    make_plot(history, 'poisson', description, destination, ylim=None)


def make_plot(history, name, description, destination, ylim=[0, 1]):
    data1 = history[name]
    data2 = history[f"val_{name}"]
    plt.figure()
    if ylim is not None:
        plt.ylim(ylim)
    plt.plot(data1, label='training')
    plt.plot(data2, label='validation')
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.title("\n".join(wrap(f"{name} - {description}", 60)))
    plt.legend()

    plt.savefig(os.path.join(destination, f"{name}.png"))


def save_model(model, destination):
    model.save_weights(os.path.join(destination, 'model.h5'))
