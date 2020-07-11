import torch
from torchvision import datasets, transforms
import params


def get_mnist(train):
    """Get MNIST dataset loader"""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.mnist_dir,
                                   train=train,
                                   transform=pre_process,
                                   download=False)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader