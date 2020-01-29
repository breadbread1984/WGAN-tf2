#!/usr/bin/python3

import tensorflow as tf;
from download_dataset import parse_function;

def Generator(input_dims = 128, inner_channels = 64):

  inputs = tf.keras.Input((128,));
  results = tf.keras.layers.Dense(units = 4 * 4 * 4 * inner_channels, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), activation = tf.keras.layers.ReLU())(inputs);
  results = tf.keras.layers.Reshape((4, 4, 4 * inner_channels))(results);
  results = tf.keras.layers.Conv2DTranspose(filters = 2 * inner_channels, kernel_size = (5,5), kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), activation = tf.keras.layers.ReLU())(results);
  results = tf.keras.layers.Lambda(lambda x: x[:,:7,:7,:])(results);
  results = tf.keras.layers.Conv2DTranspose(filters = inner_channels, kernel_size = (5,5), kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), activation = tf.keras.layers.ReLU())(results);
  results = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (8,8), kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), strides = (2,2))(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Discriminator(inner_channels = 128):

  inputs = tf.keras.Input((28,28,1));
  results = tf.keras.layers.Conv2D(filters = inner_channels, kernel_size = (5, 5), strides = (2, 2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), activation = tf.keras.layers.ReLU())(inputs);
  results = tf.keras.layers.Conv2D(filters = 2 * inner_channels, kernel_size = (5, 5), strides = (2, 2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), activation = tf.keras.layers.ReLU())(results);
  results = tf.keras.layers.Conv2D(filters = 4 * inner_channels, kernel_size = (5, 5), strides = (2, 2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02), activation = tf.keras.layers.ReLU())(results);
  results = tf.keras.layers.Flatten()(results);
  results = tf.keras.layers.Dense(units = 1, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

class WGAN(tf.keras.Model):

  def __init__(self, input_dims = 128, inner_channels = 64, **kwargs):

    super(WGAN, self).__init__(**kwargs);
    self.G = Generator(input_dims, inner_channels);
    self.D = Discriminator(inner_channels);

  def call(self, inputs):

    real = inputs;
    r = tf.random.normal((inputs.shape[0], 128), dtype = tf.float32);
    fake = self.G(r);
    real_pred = self.D(real);
    fake_pred = self.D(fake);
    return real, real_pred, fake, fake_pred;

  def loss(self, inputs):
      
    (real, real_pred, fake, fake_pred) = inputs;
    # 1) discriminator loss
    D_real = tf.math.reduce_mean(real_pred);
    D_fake = tf.math.reduce_mean(fake_pred);
    r = tf.random.uniform((real.shape[0], 1, 1, 1), dtype = tf.float32);
    interpolates = r * real + (1 - r) * fake;
    with tf.GradientTape() as g:
      g.watch(interpolates);
      D_interpolates = self.D(interpolates);
    g_D = g.gradient(D_interpolates, interpolates);
    D_loss = D_fake - D_real + 10 * (tf.norm(g_D, 2) - 1.) ** 2;
    # 2) generator loss
    G_loss = -D_fake;
    return D_loss, G_loss;

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  generator = Generator();
  discriminator = Discriminator();
  tf.keras.utils.plot_model(model = generator, to_file = 'generator.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = discriminator, to_file = 'discriminator.png', show_shapes = True, dpi = 64);

