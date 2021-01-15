import tensorflow as tf

from lib.cycleGAN.tf2lib.data import *
from lib.cycleGAN.tf2lib.image import *
from lib.cycleGAN.tf2lib.ops import *
from lib.cycleGAN.tf2lib.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
