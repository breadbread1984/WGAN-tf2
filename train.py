#!/usr/bin/python3

import os;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from download_dataset import parse_function;
from models import WGAN;

batch_size = 100;

def main():

  wgan = WGAN();
  optimizerG = tf.keras.optimizers.Adam(learning_rate = 1e-4, beta_1 = 0.5);
  optimizerD = tf.keras.optimizers.Adam(learning_rate = 1e-4, beta_1 = 0.5);
  trainset = iter(tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  checkpoint = tf.train.Checkpoint(model = wgan, optimizerG = optimizerG, optimizerD = optimizerD);
  log = tf.summary.create_file_writer('checkpoints');
  avg_d_loss = tf.keras.metrics.Mean(name = 'D loss', dtype = tf.float32);
  avg_g_loss = tf.keras.metrics.Mean(name = 'G loss', dtype = tf.float32);
  while True:
    images, _ = next(trainset);
    with tf.GradientTape(persistent = True) as tape:
      outputs = wgan(images);
      d_loss, g_loss = wgan.loss(outputs);
    d_grads = tape.gradient(d_loss, wgan.D.trainable_variables); avg_d_loss.update_state(d_loss);
    g_grads = tape.gradient(g_loss, wgan.G.trainable_variables); avg_g_loss.update_state(g_loss);
    optimizerD.apply_gradients(zip(d_grads, wgan.D.trainable_variables));
    optimizerG.apply_gradients(zip(g_grads, wgan.G.trainable_variables));
    if tf.equal(optimizerG.iterations % 100, 0):
      r = tf.random.normal((1, 128), dtype = tf.float32);
      fake = tf.clip_by_value((wgan.G(r) + 1.) * 127.5, clip_value_min = 0., clip_value_max = 255.);
      fake = tf.cast(fake, dtype = tf.uint8);
      with log.as_default():
        tf.summary.scalar('discriminator loss', avg_d_loss.result(), step = optimizerG.iterations);
        tf.summary.scalar('generator loss', avg_g_loss.result(), step = optimizerG.iterations);
        tf.summary.image('generated image', fake, step = optimizerG.iterations);
      print('Step #%d D loss: %.6f G loss: %.6f' % (optimizerG.iterations, avg_d_loss.result(), avg_g_loss.result()));
      avg_d_loss.reset_states();
      avg_g_loss.reset_states();
    if tf.equal(optimizerG.iterations % 500, 0):
      checkpoint.save(os.path.join('checkpoints','ckpt'));
  if False == os.path.exists('models'): os.mkdir('models');
  wgan.D.save(os.path.join('models','D.h5'));
  wgan.G.save(os.path.join('models','G.h5'));

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  main();
