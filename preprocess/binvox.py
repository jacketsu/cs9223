# Based on https://github.com/dimatura/binvox-rw-py

import numpy as np


class Voxel(object):

    def __init__(self, data, translate=None, scale=1.0):
        self.data = data
        self.translate = translate or [0.0, 0.0, 0.0]
        self.scale = scale

    def copy(self):
        data = self.data.copy()
        translate = self.translate[:]
        return Voxel(data, translate, self.scale)

    def write(self, fp):
        write(fp, self)


def read_header(fp):
    line = fp.readline().strip()
    if not line.startswith(b"#binvox"):
        raise IOError("Not a binvox file")
    dim = list(map(int, fp.readline().strip().split(b" ")[1:]))
    translate = list(map(float, fp.readline().strip().split(b" ")[1:]))
    scale = list(map(float, fp.readline().strip().split(b" ")[1:]))[0]
    line = fp.readline()
    return dim, translate, scale


def read(fp):
    dim, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dim)
    data = np.transpose(data, (0, 2, 1))
    return Voxel(data, translate, scale)


def write_header(fp, model):
    fp.write(b"#binvox 1\n")
    fp.write("dim {}\n".format(" ".join(map(str, model.data.shape))).encode("ascii"))
    fp.write("translate {}\n".format(" ".join(map(str, model.translate))).encode("ascii"))
    fp.write("scale {}\n".format(str(model.scale)).encode("ascii"))
    fp.write(b"data\n")


def write(fp, model):
    write_header(fp, model)
    data_flat = np.transpose(model.data, (0, 2, 1)).flatten()
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
