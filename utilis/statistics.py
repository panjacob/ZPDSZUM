import os.path

import matplotlib.pyplot as plt
from utilis.generator import create_path_if_doesnt_exist
from textwrap import wrap


def make_plots(history, description, destination):
    create_path_if_doesnt_exist(destination)
    print(history)
    make_plot(history['loss'], 'loss.png', 'loss', 'epoch', f"loss - {description}", destination)
    make_plot(history['accuracy'], 'accuracy.png', 'accuracy', 'epoch', f"accuracy - {description}", destination)
    make_plot(history['f_score'], 'f_score.png', 'f_score', 'epoch', f"f_score - {description}", destination)
    make_plot(history['AUC_ROC'], 'AUC_ROC.png', 'AUC_ROC', 'epoch', f"AUC_ROC - {description}", destination)


def make_plot(data, filename, ylabel, xlabel, title, destination):
    plt.figure()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("\n".join(wrap(title, 60)))
    plt.savefig(os.path.join(destination, filename))


def save_model(model, destination):
    model.save_weights(os.path.join(destination, 'model.h5'))
