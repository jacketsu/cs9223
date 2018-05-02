# Based on https://github.com/dimatura/binvox-rw-py

import numpy as np


class Binvox(object):

    def __init__(self, data, translate=None, scale=1.0):
        self.data = data
        self.translate = translate or [0.0, 0.0, 0.0]
        self.scale = scale

    def copy(self):
        data = self.data.copy()
        translate = self.translate[:]
        return Binvox(data, translate, self.scale)

    def toarray(self):
        return self.data

    def save(self, filename):
        with open(filename, "wb") as fp:
            # save header
            fp.write(b"#binvox 1\n")
            fp.write("dim {}\n".format(" ".join(map(str, self.data.shape))).encode("ascii"))
            fp.write("translate {}\n".format(" ".join(map(str, self.translate))).encode("ascii"))
            fp.write("scale {}\n".format(str(self.scale)).encode("ascii"))
            fp.write(b"data\n")
            # save raw data
            data_flat = np.transpose(self.data, (0, 2, 1)).flatten()
            state = data_flat[0]
            count = 0
            for d in data_flat:
                if d == state:
                    count += 1
                    if count == 255:
                        fp.write(bytes([state, count]))
                        count = 0
                else:
                    fp.write(bytes([state, count]))
                    state = d
                    count = 1
            if count > 0:
                fp.write(bytes([state, count]))


def load(filename):
    with open(filename, "rb") as fp:
        # load header
        line = fp.readline().strip()
        if not line.startswith(b"#binvox"):
            raise IOError("Not a binvox file")
        dim = list(map(int, fp.readline().strip().split(b" ")[1:]))
        translate = list(map(float, fp.readline().strip().split(b" ")[1:]))
        scale = list(map(float, fp.readline().strip().split(b" ")[1:]))[0]
        line = fp.readline()
        # load raw data
        raw_data = np.frombuffer(fp.read(), dtype="uint8")
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype("bool")
        data = data.reshape(dim)
        data = np.transpose(data, (0, 2, 1))
        return Binvox(data, translate, scale)


def fromarray(data, dim=None, threshold=0.5):
    if data.dtype != "bool":
        data = data > threshold
    if dim is not None:
        data = data.reshape(dim)
    return Binvox(data)
