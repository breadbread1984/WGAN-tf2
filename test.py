#!/usr/bin/python3

import cv2;
import tensorflow as tf;

def generate(batch_size = 1):

  G = tf.keras.models.load_model('models/G.h5', compile = False, custom_objects = {'tf': tf});
  r = tf.random.uniform((batch_size, 128), dtype = tf.float32);
  return G(r);

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  outputs = generate(batch_size = 1);
  img = tf.cast(outputs[0], dtype = tf.uint8).numpy();
  cv2.imshow('generated image', img);
  cv2.waitKey();
