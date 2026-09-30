"""MNIST hardware-aware training using a straight-through digital gradient."""
from _classification import main

if __name__ == "__main__":
    main("mlp", training=True)
