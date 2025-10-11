# ViT
A simple vision transformer implementation for image classification in python with Pytorch leading to 93.5% accuracy on FashionMNIST (test dataset) after only 100 epochs without dropout. The same model achieved the same accuracy on the MNIST dataset after only 5 epochs, and up to 99.2% after 100 epochs.

## Requirements
- Recommended, but optional : a CUDA-capable device
- `torch`, `torchvision`, `einops`

## How to adapt the code for yourself
You will need to change the dataset at the top of the code as well as some hyperparameters/values in the ViT's `__init__` method.
