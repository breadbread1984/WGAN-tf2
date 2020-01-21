#!/usr/bin/python3

import os;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from download_dataset import parse_function;
from models import WGAN;

batch_size = 100;

def main():

  wgan = WGAN();
  optimizer = tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1e-4, 0.3, 100), beta_1 = 0.5);
  trainset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False).repeat(100).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  checkpoint = tf.train.Checkpoint(model = wgan, optimizer = optimizer, optimizer_step = optimizer.iterations);
  log = tf.summary.create_file_writer('checkpoints');
  avg_d_loss = tf.keras.metrics.Mean(name = 'D loss', dtype = tf.float32);
  avg_g_loss = tf.keras.metrics.Mean(name = 'G loss', dtype = tf.float32);
  for images, _ in trainset:
    with tf.GradientTape(persistent = True) as tape:
      outputs = wgan(images);
      d_loss, g_loss = wgan.loss(outputs);
    d_grads = tape.gradient(d_loss, wgan.D.trainable_variables); avg_d_loss.update_state(d_loss);
    g_grads = tape.gradient(g_loss, wgan.G.trainable_variables); avg_g_loss.update_state(g_loss);
    optimizer.apply_gradients(zip(d_grads, wgan.D.trainable_variables));
    optimizer.apply_gradients(zip(g_grads, wgan.G.trainable_variables));
    if tf.equal(optimizer.iterations % 100, 0):
      r = tf.random.normal((1, 128), dtype = tf.float64);
      fake = wgan.G(r);
      fake = tf.cast(fake, dtype = tf.uint8);
      with log.as_default():
        tf.summary.scalar('discriminator loss', avg_d_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('generator loss', avg_g_loss.result(), step = optimizer.iterations);
        tf.summary.image('generated image', fake, step = optimizer.iterations);
      print('Step #%d D loss: %.6f G loss: %.6f' % (optimizer.iterations, avg_d_loss.result(), avg_g_loss.result()));
      avg_d_loss.reset_states();
      avg_g_loss.reset_states();
    if tf.equal(optimizer.iterations % 500, 0):
      checkpoint.save(os.path.join('checkpoints','ckpt'));
  if False == os.path.exists('models'): os.mkdir('models');
  wgan.D.save(os.path.join('models','D.h5'));
  wgan.G.save(os.path.join('models','G.h5'));

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  main();
