import os
path = 'Images/new'
files = os.listdir(path)

start = 584
for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index + start), '.jpg'])))