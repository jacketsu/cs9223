# User: shengyifan
# Date: 3/28/18
# Time: 18:42

import os
import glob
import subprocess as sub
import numpy as np
import binvox as bv

name_list = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
parameter = ["-d", "64", "-aw", "-dc", "-cb", "-pb"]

input_root = "/Users/shengyifan/Workspace/Data/ModelNet10"
output_root = input_root + "_" + parameter[1]
numpy_root = output_root + "_npy"
program_path = "/Users/shengyifan/File/NYU/2018 Spring/Deep Learning/DLP_Code/program/binvox"

os.makedirs(numpy_root)
for model_name in name_list:
    os.makedirs(os.path.join(output_root, model_name))
    models = []
    for folder_name in ["train", "test"]:
        input_path = os.path.join(input_root, model_name, folder_name, "*.off")
        input_list = glob.glob(input_path)
        input_list.sort()
        for input_file in input_list:
            print(input_file)
            sub.run([program_path] + parameter + [input_file], stdout=sub.DEVNULL, stderr=sub.DEVNULL)
            temp_file = os.path.splitext(input_file)[0] + ".binvox"
            file_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_root, model_name, file_name + ".binvox")
            os.rename(temp_file, output_file)
            with open(output_file, "rb") as fp:
                model = bv.read(fp)
                models.append(model.data.flatten())
    np.save(os.path.join(numpy_root, model_name), np.row_stack(models))


with open(os.path.join(output_root, "README.txt"), "wb") as fp:
    fp.write("binvox {}\n".format(" ".join(parameter)).encode("ascii"))

with open(os.path.join(numpy_root, "README.txt"), "wb") as fp:
    fp.write("binvox {}\n".format(" ".join(parameter)).encode("ascii"))
