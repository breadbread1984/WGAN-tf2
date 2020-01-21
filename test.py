#!/usr/bin/python3

import cv2;
import tensorflow as tf;

def generate(batch_size = 1):

  G = tf.keras.models.load_model('models/G.h5', compile = False, custom_objects = {'tf': tf, 'ReLU': tf.keras.layers.ReLU});
  r = tf.random.uniform((batch_size, 128), dtype = tf.float32);
  generated = tf.clip_by_value(G(r) * 255., clip_value_min = 0, clip_value_max = 255);
  return generated;

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  outputs = generate(batch_size = 1);
  img = tf.cast(outputs[0], dtype = tf.uint8).numpy();
  cv2.imshow('generated image', img);
  cv2.waitKey();
