import os
path = 'ZPD/Nowy'
files = os.listdir(path)

start = 466
for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index + start), '.jpg'])))