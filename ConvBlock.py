import tensorflow as tf
import keras


# Create convolution block
class ConvBlock(tf.keras.Model):
    def __init__(self, num_filters):
        super( ConvBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D( filters=num_filters,
                                              kernel_size=(3, 3),
                                              strides=(1, 1),
                                              padding='same' )
        self.activation = tf.keras.layers.ReLU()  # we can specify the activation function directly in Conv2D
        self.pooling = tf.keras.layers.MaxPool2D( pool_size=(2, 2) )

    def call(self, inputs):
        x = self.conv2d( inputs )
        x = self.activation( x )
        x = self.pooling( x )
        return x