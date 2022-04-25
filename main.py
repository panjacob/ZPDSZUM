import json
import os
import shutil
from pprint import pprint
from random import random

from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


def rl(a, b):
    return list(range(a, b + 1))


def create_path_if_doesnt_exist(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_dataset():
    path = os.path.join('files', 'classes.json')
    with open(path) as json_file:
        data = json.load(json_file)
        result = []
        for x in data:
            x['id'] = int(x['id'])
            result.append(x)
        return result


def move_images_to_folders(directory, dataset, all_path=os.path.join('images', 'All')):
    create_path_if_doesnt_exist(directory)
    for x in dataset:
        filename = str(x['id']) + '.jpg'
        class_path = os.path.join(directory, x['class'])
        create_path_if_doesnt_exist(class_path)
        shutil.copyfile(os.path.join(all_path, filename), os.path.join(class_path, filename))


def not_in_arr_arr(val, arr1):
    for arr2 in arr1:
        if val in arr2:
            return True
    return False


def generate_datasets(dataset, corelated, train_ratio, test_ratio, validation_ratio):
    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)
    validation_size = int(len(dataset) * validation_ratio)

    corelated_ids = [rl(a, b) for a, b in corelated]
    dataset_ids = list(range(1, len(dataset) + 1))
    data_set_no_corelated = [x for x in dataset_ids if not not_in_arr_arr(x, corelated_ids)]

    train_ids = []
    test_ids = []
    validation_ids = []

    for x in corelated_ids:
        random_float = random()
        if random_float <= train_ratio:
            add_to = "train"
        elif random_float <= train_ratio + test_ratio:
            add_to = "test"
        else:
            add_to = "validation"

        if add_to == "train":
            if len(train_ids) <= train_size:
                train_ids.extend(x)
            elif len(test_ids) <= test_size:
                test_ids.extend(x)
            else:
                validation_ids.extend(x)
        elif add_to == "test":
            if len(test_ids) <= test_size:
                test_ids.extend(x)
            elif len(train_ids) <= train_size:
                train_ids.extend(x)
            else:
                validation_ids.extend(x)
        else:
            if len(validation_ids) <= validation_size:
                validation_ids.extend(x)
            elif len(train_ids) <= train_size:
                train_ids.extend(x)
            else:
                test_ids.extend(x)

    for x in data_set_no_corelated:
        random_float = random()
        if random_float <= train_ratio:
            add_to = "train"
        elif random_float <= train_ratio + test_ratio:
            add_to = "test"
        else:
            add_to = "validation"

        if add_to == "train":
            if len(train_ids) <= train_size:
                train_ids.append(x)
            elif len(test_ids) <= test_size:
                test_ids.append(x)
            else:
                validation_ids.append(x)
        elif add_to == "test":
            if len(test_ids) <= test_size:
                test_ids.append(x)
            elif len(train_ids) <= train_size:
                train_ids.append(x)
            else:
                validation_ids.append(x)
        else:
            if len(validation_ids) <= validation_size:
                validation_ids.append(x)
            elif len(train_ids) <= train_size:
                train_ids.append(x)
            else:
                test_ids.append(x)

    train_set = [x for x in dataset if x['id'] in train_ids]
    test_set = [x for x in dataset if x['id'] in test_ids]
    validation_set = [x for x in dataset if x['id'] in test_ids]

    return train_set, test_set, validation_set


keras_dir = os.path.join('images', 'keras')
train_dir = os.path.join(keras_dir, 'train')
test_dir = os.path.join(keras_dir, 'test')
validation_dir = os.path.join(keras_dir, 'validation')

dataset = load_dataset()
corelated = [(1, 20), (21, 83), (84, 92), (93, 114), (115, 209), (210, 227), (228, 262), (283, 366),
             (420, 443), (444, 465)]

train_set, test_set, validation_set = generate_datasets(dataset, corelated, 0.6, 0.2, 0.2)
move_images_to_folders(train_dir, train_set)
move_images_to_folders(test_dir, test_set)
move_images_to_folders(validation_dir, validation_set)

datagen = ImageDataGenerator(
    rescale=(1.0 / 255.0),
    featurewise_center=True,
    featurewise_std_normalization=True)

train_generator = datagen.flow_from_directory(train_dir)
test_generator = datagen.flow_from_directory(test_dir)
validation_generator = datagen.flow_from_directory(validation_dir)
