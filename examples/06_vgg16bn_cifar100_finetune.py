"""Fine-tune the published CIFAR100 VGG16-BN with simulated forward operations."""
from _classification import main

if __name__ == "__main__":
    main("vgg_cifar100", training=True)
