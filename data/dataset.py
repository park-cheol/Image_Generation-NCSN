import os

from torchvision.datasets import MNIST, CIFAR10, SVHN

from data.celebA import CelebA


def get_dataset(dataset_name, train_transform, test_transform):
    if dataset_name == 'cifar10':
        print("dataset : CIFAR 10")
        train_dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                                transform=train_transform)
        test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)
    elif dataset_name == 'mnist':
        print("dataset : MNIST")
        train_dataset = MNIST(os.path.join('datasets', 'minst'), train=True, download=True,
                              transform=train_transform)
        test_dataset = MNIST(os.path.join('datasets', 'minst_test'), train=False, download=True,
                             transform=test_transform)
    elif dataset_name == 'celeba':
        print("dataset : celeba")
        train_dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                               transform=train_transform,
                               download=True)
        test_dataset = CelebA(root=os.path.join('datasets', 'celeba_test'), split='test',
                              transform=test_transform,
                              download=True)
    else:
        raise Exception('Dataset Error')

    return train_dataset, test_dataset

