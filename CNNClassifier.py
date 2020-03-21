import tensorflow as tf
from ConvBlock import ConvBlock
import keras
# Create Model
# ------------

#depth = 5
#start_f = 8
#num_classes = 20


class CNNClassifier( tf.keras.Model ):
    def __init__(self, depth, start_f, num_classes):
        super( CNNClassifier, self ).__init__()

        self.feature_extractor = tf.keras.Sequential()

        for i in range( depth ):
            self.feature_extractor.add( ConvBlock( num_filters=start_f ) )
            start_f *= 2

        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential()
        self.classifier.add( tf.keras.layers.Dense( units=512, activation='relu' ) )
        self.classifier.add( tf.keras.layers.Dense( units=num_classes, activation='softmax' ) )

    def call(self, inputs):
        x = self.feature_extractor( inputs )
        x = self.flatten( x )
        x = self.classifier( x )
        return x


## Create Model instance
#model = CNNClassifier( depth=depth,
#                       start_f=start_f,
#                       num_classes=num_classes )
## Build Model (Required)
#model.build( input_shape=(None, img_h, img_w, 3) )