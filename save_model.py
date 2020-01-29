#!/usr/bin/python3

import os;
import tensorflow as tf;
from models import WGAN;

def save_model():

  wgan = WGAN();
  optimizerG = tf.keras.optimizers.Adam(2e-4);
  optimizerD = tf.keras.optimizers.Adam(2e-4);
  checkpoint = tf.train.Checkpoint(model = wgan, optimizerG = optimizerG, optimizerD = optimizerD);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == os.path.exists('models'): os.mkdir('models');
  wgan.G.save(os.path.join('models', 'G.h5'));
  wgan.D.save(os.path.join('models', 'D.h5'));

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  save_model();
