"""Distributed MNIST hardware-aware training; also accepts a single process."""
from _classification import main

if __name__ == "__main__":
    main("mlp", training=True, distributed=True)
