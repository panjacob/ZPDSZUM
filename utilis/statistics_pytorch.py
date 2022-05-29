import os
from utilis.generator import create_path_if_doesnt_exist
from textwrap import wrap
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np


def make_confusion_matrix(loader, model, class_number):
    y_pred = []
    y_true = []
    # iterate over test data
    for inputs, labels in loader:
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes4
    if class_number == 6:
        classes = ('Front', 'Front_angle', 'Others', 'Rear', 'Rear_angle',
                'Side')
    elif class_number == 4:
        classes = ('Front', 'Others', 'Rear',
                'Side')
    else:
        print("Liczba klas powinna wynosić 6 lub 4.")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *4, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')

def make_plots(history, description, destination):
    create_path_if_doesnt_exist(destination)
    print(history)
    make_plot(history, 'loss', description, destination)
    make_plot(history, 'accuracy', description, destination)



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