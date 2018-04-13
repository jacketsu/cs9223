import numpy as np
import tensorflow as tf


data_path = "/Users/shengyifan/Workspace/Data/ModelNet10_64_npy/bathtub.npy"
bathtub = np.load(data_path).astype(np.float32)


def get_train_input_fn(batch_size, noise_dims):
    def train_input_fn():
        voxels = tf.estimator.inputs.numpy_input_fn(x=bathtub,
                                                    y=None,
                                                    batch_size=batch_size,
                                                    num_epochs=None,
                                                    shuffle=True)()
        noise = tf.random_uniform([batch_size, noise_dims])
        return noise, voxels
    return train_input_fn


def get_predict_input_fn(batch_size, noise_dims):
    def predict_input_fn():
        noise = tf.random_uniform([batch_size, noise_dims])
        return noise
    return predict_input_fn
