# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Networks for 3D Object using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan


def _generator_helper(
    noise, is_conditional, one_hot_labels, weight_decay, is_training):
  """Core 3D  generator.
  This function is reused between the different GAN models (unconditional,
  conditional, etc).
  """
  with tf.contrib.framework.arg_scope(
      [layers.conv3d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      #print(noise.shape)
      #net = layers.fully_connected(noise, 2048)
      #if is_conditional:
       # net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
      #net = tf.reshape(noise, [-1, 1, 1, 1, 200])
      #net = layers.fully_connected(net, 4 * 4 * 4 * 200)
      net = tf.reshape(noise, [-1, 1, 1, 1, 200])
      #print(net.shape)
      net = layers.conv3d_transpose(net, 512, kernel_size=4, stride=1, padding="VALID")
      #print(net.shape)
      net = layers.conv3d_transpose(net, 256, kernel_size=4, stride=2)
      net = layers.conv3d_transpose(net, 128, kernel_size=4, stride=2)
      net = layers.conv3d_transpose(net, 64, kernel_size=4, stride=2)
      #print(net.shape)
      # Make sure that generator output is in the same range as `inputs`
      # ie [-1, 1].
      net = layers.conv3d_transpose(net, 1, kernel_size=4, stride=2, normalizer_fn=None, activation_fn=tf.sigmoid)
     # print(net.shape)
      return net


def unconditional_generator(noise, weight_decay=2.5e-5, is_training=True):
  """Generator to produce unconditional 3D objects.
  """
  return _generator_helper(noise, False, None, weight_decay, is_training)


def conditional_generator(inputs, weight_decay=2.5e-5, is_training=True):
  noise, one_hot_labels = inputs
  return _generator_helper(
      noise, True, one_hot_labels, weight_decay, is_training)


def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5,
                      is_training=True):
  unstructured_noise, cat_noise, cont_noise = inputs
  cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
  all_noise = tf.concat(
      [unstructured_noise, cat_noise_onehot, cont_noise], axis=1)
  return _generator_helper(all_noise, False, None, weight_decay, is_training)


_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.2)


def _discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
  """Core 3D  discriminator.
  This function is reused between the different GAN modes (unconditional,
  conditional, etc).
  Args:
    img: Real or generated 3D digits. Should be in the range [-1, 1].
    is_conditional: Whether to condition on labels.
    one_hot_labels: Labels to optionally condition the network on.
    weight_decay: The L2 weight decay.
  Returns:
    Final fully connected discriminator layer. [batch_size, 1024].
  """
  with tf.contrib.framework.arg_scope(
      [layers.conv3d],
      activation_fn=_leaky_relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay),
      biases_regularizer=layers.l2_regularizer(weight_decay)): 
     #print(img.shape)
     #net = tf.reshape(img, [-1, 16, 16, 16, 1])
     net = layers.conv3d(img, 64, kernel_size=4, stride=2)
     net = layers.conv3d(net, 128, kernel_size=4, stride=2)
     net = layers.conv3d(net, 256, kernel_size=4, stride=2)
     net = layers.conv3d(net, 512, kernel_size=4, stride=2)
     #print(net.shape)
     net = layers.conv3d(net, 1, kernel_size=4, stride=1, padding="VALID", activation_fn=tf.sigmoid)
     #net = layers.flatten(net)
     #print(net.shape)
# if is_conditional:
     # net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
     #net = layers.fully_connected(net, 2048, normalizer_fn=layers.layer_norm)
     #print(net.shape)
     return net


def unconditional_discriminator(img, unused_conditioning, weight_decay=2.5e-5, is_training=True):
 # print(img.shape)
  net = _discriminator_helper(img, False, None, weight_decay)
  return layers.linear(net, 1)


def conditional_discriminator(img, conditioning, weight_decay=2.5e-5):
  
  _, one_hot_labels = conditioning
  net = _discriminator_helper(img, True, one_hot_labels, weight_decay)
  return layers.linear(net, 1)


def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                          categorical_dim=10, continuous_dim=2):
  
  
  net = _discriminator_helper(img, False, None, weight_decay)
  logits_real = layers.fully_connected(net, 1, activation_fn=None)

  # Recognition network for latent variables has an additional layer
  with tf.contrib.framework.arg_scope([layers.batch_norm], is_training=False):
    encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm,
                                     activation_fn=_leaky_relu)

  # Compute logits for each category of categorical latent.
  logits_cat = layers.fully_connected(
      encoder, categorical_dim, activation_fn=None)
  q_cat = ds.Categorical(logits_cat)

  # Compute mean for Gaussian posterior of continuous latents.
  mu_cont = layers.fully_connected(encoder, continuous_dim, activation_fn=None)
  sigma_cont = tf.ones_like(mu_cont)
  q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

  return logits_real, [q_cat, q_cont]