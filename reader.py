import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import struct as st


class Reader:

    def __init__(self, root_dir):
        self.root_dir = root_dir

    @staticmethod
    def plot_images(in_data, n, random=False):
        data = np.array([d for d in in_data])
        data = data.reshape(data.shape[0], 28, 28).astype("uint8")
        x1 = min(n // 2, 5)  # Floor Division
        if x1 == 0:
            x1 = 1
        y1 = (n // x1)
        x = min(x1, y1)
        y = max(x1, y1)
        fig, ax = plt.subplots(x, y, figsize=(5, 5))
        i = 0
        for j in range(x):
            for k in range(y):
                if random:
                    i = np.random.choice(range(len(data)))
                ax[j][k].set_axis_off()
                ax[j][k].imshow(data[i:i + 1][0])
                i += 1
        plt.show()

    def get_data(self, data_folder='train'):
        data_types = {
            0x08: ('ubyte', 'B', 1),
            0x09: ('byte', 'b', 1),
            0x0B: ('>i2', 'h', 2),
            0x0C: ('>i4', 'i', 4),
            0x0D: ('>f4', 'f', 4),
            0x0E: ('>f8', 'd', 8)}

        images_file = open(join(self.root_dir, data_folder + '/train-images.idx3-ubyte'), 'rb')
        labels_file = open(join(self.root_dir, data_folder + '/train-labels.idx1-ubyte'), 'rb')

        images_file.seek(0)  # Sets file buffer t first

        # Read 4 bytes to check incoming file format
        magic = st.unpack('>4B', images_file.read(4))
        if (magic[0] and magic[1]) or (magic[2] not in data_types):
            raise ValueError("File Format not correct")

        images_file.seek(4)

        # Get DataSet Info
        images_count = st.unpack('>I', images_file.read(4))[0]
        rows_count = st.unpack('>I', images_file.read(4))[0]
        columns_count = st.unpack('>I', images_file.read(4))[0]
        total_bytes_count = images_count * rows_count * columns_count

        # Time to Read DataSet Content
        labels_file.seek(8)
        images_array = 255 - np.asarray(
            st.unpack('>' + 'B' * total_bytes_count, images_file.read(total_bytes_count))).reshape(
            (images_count, rows_count, columns_count))
        labels_array = np.asarray(
            st.unpack('>' + 'B' * images_count, labels_file.read(images_count))).reshape((images_count, 1))
        labels_array = [lbl[0] for lbl in labels_array]

        return images_array.reshape(60000, 28 * 28), labels_array
