import tensorflow as tf


def generator_fn(noise):

    g_input = tf.reshape(noise, [-1, 1, 1, 1, 200], name="g_input")

    g1 = tf.layers.conv3d_transpose(inputs=g_input,
                                    filters=512,
                                    kernel_size=4,
                                    strides=1,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    name="g1")

    g2 = tf.layers.conv3d_transpose(inputs=g1,
                                    filters=256,
                                    kernel_size=4,
                                    strides=2,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    name="g2")

    g3 = tf.layers.conv3d_transpose(inputs=g2,
                                    filters=128,
                                    kernel_size=4,
                                    strides=2,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    name="g3")

    g4 = tf.layers.conv3d_transpose(inputs=g3,
                                    filters=64,
                                    kernel_size=4,
                                    strides=2,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    name="g4")

    g5 = tf.layers.conv3d_transpose(inputs=g4,
                                    filters=1,
                                    kernel_size=4,
                                    strides=2,
                                    padding="same",
                                    activation=tf.nn.sigmoid,
                                    name="g5")

    return g5


def discriminator_fn(voxels, unused_conditioning):

    d_input = tf.reshape(voxels, [-1, 64, 64, 64, 1], name="d_input")

    d1 = tf.layers.conv3d(inputs=d_input,
                          filters=64,
                          kernel_size=4,
                          strides=2,
                          padding="same",
                          activation=tf.nn.leaky_relu,
                          name="d1")

    d2 = tf.layers.conv3d(inputs=d1,
                          filters=128,
                          kernel_size=4,
                          strides=2,
                          padding="same",
                          activation=tf.nn.leaky_relu,
                          name="d2")

    d3 = tf.layers.conv3d(inputs=d2,
                          filters=256,
                          kernel_size=4,
                          strides=2,
                          padding="same",
                          activation=tf.nn.leaky_relu,
                          name="d3")

    d4 = tf.layers.conv3d(inputs=d3,
                          filters=512,
                          kernel_size=4,
                          strides=2,
                          padding="same",
                          activation=tf.nn.leaky_relu,
                          name="d4")

    d5 = tf.layers.conv3d(inputs=d4,
                          filters=64,
                          kernel_size=4,
                          strides=1,
                          padding="valid",
                          activation=tf.nn.sigmoid,
                          name="d5")

    return d5

