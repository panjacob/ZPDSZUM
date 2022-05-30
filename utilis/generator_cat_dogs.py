# import os
# import shutil
#
# from utilis.generator import create_path_if_doesnt_exist
#
#
# def create_directories_dataset(dataset_name):
#     dataset_dir = os.path.join('..', 'images', dataset_name)
#     train_dir = os.path.join(dataset_dir, 'train')
#     test_dir = os.path.join(dataset_dir, 'test')
#     is_created = not os.path.exists(dataset_dir)
#
#     for x in [dataset_dir, train_dir, test_dir]:
#         create_path_if_doesnt_exist(x)
#
#     return dataset_dir, train_dir, test_dir, is_created
#
#
# dataset_dir, train_dir, test_dir, is_created = create_directories_dataset(os.path.join('cat_dog'))
# path = os.path.join('..', 'old', 'cat_dogs', 'train', 'train')
# files = os.listdir(path)
#
# files_len = len(files)
# create_path_if_doesnt_exist(os.path.join(test_dir, 'dog'))
# create_path_if_doesnt_exist(os.path.join(test_dir, 'cat'))
# create_path_if_doesnt_exist(os.path.join(train_dir, 'cat'))
# create_path_if_doesnt_exist(os.path.join(train_dir, 'dog'))
# for index, file in enumerate(files):
#
#     if file.split('.')[0] == 'dog':
#         shutil.copyfile(os.path.join(path, file), os.path.join(train_dir, 'dog', file))
#     else:
#         print('else')
#         shutil.copyfile(os.path.join(path, file), os.path.join(train_dir, 'cat', file))
# else:
#     if file.__contains__('dog'):
#         shutil.copyfile(os.path.join(path, file), os.path.join(test_dir, 'dog', file))
#     else:
#         shutil.copyfile(os.path.join(path, file), os.path.join(test_dir, 'cat', file))
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generators(dataset_name, image_width=256):
    dataset_dir = os.path.join('images', dataset_name)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    datagen = ImageDataGenerator(
        rescale=(1.0 / 255.0),
        featurewise_center=False,
        featurewise_std_normalization=False,
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.6, 1.4],
        # channel_shift_range=40,
        zoom_range=0.2,
    )

    train_generator = datagen.flow_from_directory(train_dir)
    test_generator = datagen.flow_from_directory(test_dir)
    if image_width != 256:
        train_generator = datagen.flow_from_directory(train_dir, target_size=(image_width, image_width))
        test_generator = datagen.flow_from_directory(test_dir, target_size=(image_width, image_width))
    return train_generator, test_generator
