import tensorflow as tf

import data
import network


tfgan = tf.contrib.gan
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    config = tf.estimator.RunConfig(model_dir="model", log_step_count_steps=10)
    gan_estimator = tfgan.estimator.GANEstimator(generator_fn=network.generator_fn,
                                                 discriminator_fn=network.discriminator_fn,
                                                 generator_loss_fn=tfgan.losses.modified_generator_loss,
                                                 discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
                                                 generator_optimizer=tf.train.AdamOptimizer(learning_rate=0.0025, beta1=0.5),
                                                 discriminator_optimizer=tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.5),
                                                 config=config)

    train_input_fn = data.get_train_input_fn(batch_size=100, noise_dims=200)
    gan_estimator.train(input_fn=train_input_fn, steps=1000)


if __name__ == "__main__":
    tf.app.run()
