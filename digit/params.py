import argparse

gpu_id = '0'
s = 0
t = 1

epochs = 30
batch_size = 64
class_num = 10
mode = 'm2u'
lr = 0.01
seed = 2020
layer = "wn"
bottleneck = 256
classifier = "bn"
smooth = 0.1

dataset_mean = 0.5
dataset_std = 0.5

data_root = "data"
usps_dir = "./data/usps"
mnist_dir = "./data/mnist"

output_dir = "results"