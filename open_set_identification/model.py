
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation
from tensorflow.keras import Model

#NEW MODEL BUILDING
class Open_ANN(Model):
  
    def __init__(self):
        super(Open_ANN, self).__init__()
        
        with tf.name_scope('representation'):
            self.conv1 = Conv2D(32,3, use_bias = True, name = 'conv1')
            self.bn1 = BatchNormalization(name = 'bn1')
        
            self.conv2 = Conv2D(64,3, use_bias = True, name = 'conv2')
            self.bn2 = BatchNormalization(name = 'bn2')
        
            self.flatten = Flatten(name = 'flatten')
            self.d1 = Dense(128, name = 'Dense1')
            self.bn3 = BatchNormalization(name = 'bn3')
        
            self.d2 = Dense(6, name = 'representation_layer')
            self.bn4 = BatchNormalization(name = 'bn4')

    def __call__(self, x,training = False):
        z = self.conv1(x)
        z = self.bn1(z, training = training)
        z = tf.nn.relu(z)

        z = self.conv2(z)
        z = self.bn2(z, training = training)
        z = tf.nn.relu(z)

        z = self.flatten(z)
        z = self.d1(z)
        z = self.bn3(z, training = training)
        z = tf.nn.relu(z)

        z = self.d2(z)
        z = self.bn4(z, training = training)

        return z

class classificationNN(Model):

    def __init__(self):
        super(classificationNN, self).__init__()

        with tf.name_scope('classification'):
            self.conv1 = Conv2D(32,3, use_bias = True, name = 'conv1')
            self.bn1 = BatchNormalization(name = 'bn1')

            self.flatten = Flatten(name = 'flatten')

            self.d1 = Dense(6, name = 'classification_layer')
            self.bn2 = BatchNormalization(name = 'bn5')

    def call(self, x, training = False):

        z = self.conv1(x)
        z = self.bn1(z, training = training)
        tf.nn.relu(z)

        z = self.flatten(z)

        z = self.d1(z)
        z = self.bn2(z, training = training)
        z = tf.nn.softmax(z)

        return z


