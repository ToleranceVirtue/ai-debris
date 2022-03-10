#! -*- coding: utf-8 -*-

import tensorflow as tf

print("gpu_available:", tf.test.is_gpu_available())

print("device_name:", tf.test.gpu_device_name())