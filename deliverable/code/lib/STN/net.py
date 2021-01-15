import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model
from lib.biometrics import extract_features

class STNet(Model):

    def __init__(self):
        super(STNet, self).__init__()

        ## 1. Localisation network
        # use MLP as the localisation net
        self.flatten1 = Flatten()
        self.dense1 = Dense(n_units=20, in_channels=30000, act=tf.nn.tanh)
        self.dropout1 = Dropout(keep=0.8)
        # you can also use CNN instead for MLP as the localisation net

        ## 2. Spatial transformer module (sampler)
        self.stn = SpatialTransformer2dAffine(out_size=(300, 100), in_channels=20)

        ## 3. Classifier
        self.conv1 = Conv2d(16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', in_channels=1)
        self.conv2 = Conv2d(16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', in_channels=16)
        self.flatten2 = Flatten()
        self.dense2 = Dense(n_units=1024, in_channels=30000, act=tf.nn.relu)
        self.dense3 = Dense(n_units=123, in_channels=1024, act=tf.identity)

    def forward(self, inputs):
        # inputs = extract_features(inputs.numpy()[:,:,0])
        theta_input = self.dropout1(self.dense1(self.flatten1(inputs)))
        V = self.stn((theta_input, inputs))
        _logits = self.dense3(self.dense2(self.flatten2(self.conv2(self.conv1(V)))))
        return _logits, V
    
    def classify(self, input):
        return self.dense3(self.dense2(self.flatten2(self.conv2(self.conv1(input)))))

