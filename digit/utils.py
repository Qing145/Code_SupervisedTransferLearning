import torch
import params
from datasets.mnist import get_mnist
from datasets.usps import get_usps


def load_data(mode, train=True):
    if mode == 's2m':
        return get_mnist(train), get_usps(train)
    elif mode == 'u2m':
        return get_mnist(train), get_usps(train)
    elif mode == 'm2u':
        return get_mnist(train), get_usps(train)
