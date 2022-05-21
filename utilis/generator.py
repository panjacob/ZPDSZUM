import json
import os
import shutil
from dataclasses import dataclass
from pprint import pprint
from random import random

from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

import os
from keras.preprocessing.image import ImageDataGenerator

ALL_IMAGES_PATH = os.path.join('images', 'All')


# is_create_new_dataset - Ustawić raz, żeby się wygenerował a potem zmienić na False, żeby za każdym razem nie losować nowych danych
def get_generators(dataset_name, train_size, image_width=256):
    dataset_dir, train_dir, test_dir, is_created = create_directories_dataset(dataset_name)
    print(is_created)
    if is_created:
        print('Tworzenie nowego datasetu i zapisywanie go w pliku: ', dataset_dir)
        dataset = load_dataset()
        corelated = [(1, 20), (21, 83), (84, 92), (93, 114), (115, 209), (210, 227), (228, 262), (283, 366),
                     (420, 443), (444, 465)]

        train_set, test_set = generate_datasets(dataset, corelated, train_size)

        move_images_to_folders(train_dir, train_set)
        move_images_to_folders(test_dir, test_set)

    datagen = ImageDataGenerator(
        rescale=(1.0 / 255.0),
        featurewise_center=False,
        featurewise_std_normalization=False)

    train_generator = datagen.flow_from_directory(train_dir)
    test_generator = datagen.flow_from_directory(test_dir)
    if image_width != 256:
        train_generator = datagen.flow_from_directory(train_dir, target_size=(image_width, image_width))
        test_generator = datagen.flow_from_directory(test_dir, target_size=(image_width, image_width))
    return train_generator, test_generator


def get_generators_augumented(dataset_name, train_size, image_width=256):
    dataset_dir, train_dir, test_dir, is_created = create_directories_dataset(dataset_name)
    print(is_created)
    if is_created:
        print('Tworzenie nowego datasetu i zapisywanie go w pliku: ', dataset_dir)
        dataset = load_dataset()
        corelated = [(1, 20), (21, 83), (84, 92), (93, 114), (115, 209), (210, 227), (228, 262), (283, 366),
                     (420, 443), (444, 465)]

        train_set, test_set = generate_datasets(dataset, corelated, train_size)

        move_images_to_folders(train_dir, train_set)
        move_images_to_folders(test_dir, test_set)

    datagen = ImageDataGenerator(
        rescale=(1.0 / 255.0),
        featurewise_center=False,
        featurewise_std_normalization=False,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.6, 1.4],
        zoom_range=[0.5, 1.0],
    )

    train_generator = datagen.flow_from_directory(train_dir)
    test_generator = datagen.flow_from_directory(test_dir)
    if image_width != 256:
        train_generator = datagen.flow_from_directory(train_dir, target_size=(image_width, image_width))
        test_generator = datagen.flow_from_directory(test_dir, target_size=(image_width, image_width))
    return train_generator, test_generator


def rl(a, b):
    return list(range(a, b + 1))


def create_path_if_doesnt_exist(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def create_directories_dataset(dataset_name):
    dataset_dir = os.path.join('images', dataset_name)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    is_created = not os.path.exists(dataset_dir)

    for x in [dataset_dir, train_dir, test_dir]:
        create_path_if_doesnt_exist(x)

    return dataset_dir, train_dir, test_dir, is_created


def load_dataset():
    path = os.path.join('files', 'classes.json')
    with open(path) as json_file:
        data = json.load(json_file)
        result = []
        for x in data:
            x['id'] = int(x['id'])
            result.append(x)
        return result


def findCar(dataset, idx):
    for i, car in enumerate(dataset):
        # print(car['id'], idx, type(car['id']), type(idx))
        if car['id'] == idx:
            return i
    return None


def get_id_from_filename(filename):
    if "_" not in filename:
        return int(filename.split('.')[0])
    return int(filename.split("_")[0])


def createCars(dataset, files):
    for i, file in enumerate(files):
        idx = get_id_from_filename(file)
        index = findCar(dataset, idx)
        if index is None:
            continue
        dataset[index]['images'].append(file)
    return dataset


def move_images_to_folders(directory, dataset):
    create_path_if_doesnt_exist(directory)
    files = os.listdir(ALL_IMAGES_PATH)
    cars = createCars(dataset, files)
    for car in cars:
        for filename in car['images']:
            class_path = os.path.join(directory, car['class'])
            create_path_if_doesnt_exist(class_path)
            shutil.copyfile(os.path.join(ALL_IMAGES_PATH, filename), os.path.join(class_path, filename))


def not_in_arr_arr(val, arr1):
    for arr2 in arr1:
        if val in arr2:
            return True
    return False


def generate_datasets(dataset, corelated, train_ratio):
    test_ratio = 1 - train_ratio
    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)

    corelated_ids = [rl(a, b) for a, b in corelated]
    dataset_ids = list(range(1, len(dataset) + 1))
    data_set_no_corelated = [x for x in dataset_ids if not not_in_arr_arr(x, corelated_ids)]

    train_ids = []
    test_ids = []

    for x in corelated_ids:
        random_float = random()
        add_to = random_float <= train_ratio

        if add_to:
            if len(train_ids) <= train_size:
                train_ids.extend(x)
            else:
                test_ids.extend(x)

        else:
            if len(test_ids) <= test_size:
                test_ids.extend(x)
            else:
                train_ids.extend(x)

    for x in data_set_no_corelated:
        random_float = random()
        add_to = random_float <= train_ratio

        if add_to:
            if len(train_ids) <= train_size:
                train_ids.append(x)
            else:
                test_ids.append(x)

        else:
            if len(test_ids) <= test_size:
                test_ids.append(x)
            else:
                train_ids.append(x)

    train_set = [x for x in dataset if x['id'] in train_ids]
    test_set = [x for x in dataset if x['id'] in test_ids]

    return train_set, test_set
