#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def parse_function(feature):

  data = feature["image"];
  data = tf.cast(data, dtype = tf.float32) / 127.5 - 1.;
  label = feature["label"];
  return data, label;
  
def download_mnist():

  mnist_builder = tfds.builder("mnist");
  mnist_builder.download_and_prepare();
  mnist_train = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
  mnist_test = tfds.load(name = "mnist", split = tfds.Split.TEST, download = False);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  download_mnist();

