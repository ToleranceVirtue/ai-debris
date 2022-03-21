#! -*- coding: utf-8 -*-

import tensorflow as tf
from tqdm import tqdm
from time import sleep

print("gpu_available:", tf.test.is_gpu_available())

print("device_name:", tf.test.gpu_device_name())

n_iter = 300
tbar = tqdm(position=0, leave=True, bar_format="{desc}")
pbar = tqdm(range(n_iter), total=n_iter, position=1, leave=True, miniters=1)

x = 0
while x < n_iter:
    x += 1
    tbar.set_description(f"Run #{x}")
    pbar.set_description("Preprocessing")
    # sleep(0.5)
    pbar.set_description("Executing")
    # sleep(0.5)
    pbar.set_description("Postprocessing")
    pbar.update()
pbar.set_description("Computation finished")