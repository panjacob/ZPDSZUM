import numpy
import numpy as np
import skimage
import matplotlib.pyplot as plt

import os

path = 'Images/All'
files = os.listdir(path)

colors = ("red", "green", "blue")
bin_edges = None
channel_ids = (0, 1, 2)
plt.figure()
plt.xlim([0, 256])
histograms = np.zeros((3, 256), dtype=numpy.int64)
files_len = len(files)
for index, file in enumerate(files):
    print(f"{index+1}/{files_len}")
    image = skimage.io.imread(os.path.join(path, file))
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256))
        for i in range(0, 256):
            histograms[channel_id][i] += histogram[i]

for channel_id in channel_ids:
    for i in range(0, 256):
        histograms[channel_id][i] = histograms[channel_id][i] / len(files)

for channel_id, c in zip(channel_ids, colors):
    plt.plot(bin_edges[0:-1], histograms[channel_id], color=c)
plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

plt.show()
