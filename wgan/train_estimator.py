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
"""Trains a GANEstimator on 3D Object data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import scipy.misc
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import networks

tfgan = tf.contrib.gan
flags = tf.flags

flags.DEFINE_integer('batch_size', 100,
                     'The number of images in each train batch.')

flags.DEFINE_integer('max_number_of_steps', 5000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'noise_dims', 200, 'Dimensions of the generator noise vector')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('eval_dir', './tmp/mnist-estimator/',
                    'Directory where the results are saved to.')

flags.DEFINE_string('train_log_dir', './tmp_sig/log/',
                    'Directory where to write event logs.')

FLAGS = flags.FLAGS

img = np.load("chair.npy").astype(np.float32).reshape(-1, 64, 64, 64, 1)
def _get_train_input_fn(batch_size,  noise_dims, img):
  def train_input_fn():
    voxels = tf.estimator.inputs.numpy_input_fn(x={"x":img},
                                                y=None,
                                                batch_size=batch_size,
                                                num_epochs=None,
                                                shuffle=True)() 
    noise = tf.random_normal([batch_size, noise_dims])
    voxel = voxels["x"]
    return noise, voxel
  return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
  def predict_input_fn():
    noise = tf.random_normal([batch_size, noise_dims])
     
    return noise
  return predict_input_fn


def _unconditional_generator(noise, mode):
  """3D generator with extra argument for tf.Estimator's `mode`."""
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  return networks.unconditional_generator(noise, is_training=is_training)


def main(_):
  # Initialize GANEstimator with options and hyperparameters.
  config = tf.estimator.RunConfig(model_dir=FLAGS.train_log_dir, save_summary_steps=10, keep_checkpoint_max=1)
  gan_estimator = tfgan.estimator.GANEstimator(
      generator_fn=_unconditional_generator,
      discriminator_fn=networks.unconditional_discriminator,
      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      generator_optimizer=tf.train.RMSPropOptimizer(0.0025),
      discriminator_optimizer=tf.train.RMSpropOptimizer(0.00001),
      config=config,
      add_summaries=tfgan.estimator.SummaryType.VARIABLES)

  # Train estimator.
  train_input_fn = _get_train_input_fn(
      FLAGS.batch_size, FLAGS.noise_dims, img)
  #print(train_input_fn)
  gan_estimator.train(train_input_fn, max_steps=FLAGS.max_number_of_steps)
  
  # Run inference.
  predict_input_fn = _get_predict_input_fn(10, FLAGS.noise_dims)
  prediction_iterable = gan_estimator.predict(predict_input_fn, hooks=[tf.train.StopAtStepHook(last_step=1)])
  predictions = [next(prediction_iterable).flatten() for _ in range(10)]
 
  np.save("chair_result", np.row_stack(predictions))


if __name__ == '__main__':
  tf.app.run()
