import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.activations import linear, relu, sigmoid
from autils import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


plt_act_trio()

_ = plt_relu_ex()