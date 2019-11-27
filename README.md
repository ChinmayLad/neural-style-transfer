# Neural Style Transfer

This is a simple repository that implements **Multiple Style Transfer Network** mentioned in paper [A Learned Representation For Artistic Style](https://arxiv.org/abs/1603.08155).

## Requirement

* PyTorch
* torchvision
* OpenCV (cv2)
* Numpy

## Conditional Instance Normalization
PyTorch does not contain api for Conditional Instance Normalization.
This is a implementation that uses code from torch [BatchNorm](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html) and is
tweaked as to have condition learnable weights.

The idea is to have weight tensor of size L x C, where L is the no. of
style representations during training.

During the forward pass we provide input image as well as style condition "label" ~ \[0,L).
During backward pass only `weigth[label:]` tensor are updated and rest remaing the same.
Here weight and bias refer to ![\gamma](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cgamma) and ![\beta](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cbeta) used for normalization.

## Result
<img src="/video/input.gif" width="30%" height="30%">  <img src="/video/output5.gif" width="30%" height="30%">
