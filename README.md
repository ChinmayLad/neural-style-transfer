# Neural Style Transfer

This is a simple implementation **Multiple Style Transfer Network** mentioned in paper [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629). The network is trained on COCO Dataset containing 80K images for 40K iteration. However we can use any dataset to while testing the style-transfer model. The model is trained using 6 style images taken from WikiArt Dataset provided by [ArtGAN repo](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md).

Here's a [blog](https://medium.com/analytics-vidhya/understanding-neural-style-transfer-3061cd92648?) I wrote that describes the mathematics for neural style transfer in detail.

## Requirement

* PyTorch
* torchvision
* OpenCV (cv2)
* Numpy
* PIL

## Conditional Instance Normalization
PyTorch does not contain api for Conditional Instance Normalization.
This is a implementation that uses code from torch [BatchNorm](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html) and is
tweaked as to have condition learnable weights.

The idea is to have weight tensor of size L x C, where L is the no. of
style representations during training.

During the forward pass we provide input image as well as style condition "label" ~ \[0,L).
During backward pass only `weigth[label:]` tensor are updated and rest remaing the same.
Here weight and bias refer to ![\gamma](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cgamma) and ![\beta](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cbeta) used for normalization.

## Results
Test images from COCO dataset.
<img src="/demo/image_table.png">

The model also runs good on CelebA dataset which was not used for training.
<img src="/demo/celeb_image.png">

The model is fast enough to convert video into Artistic Styles in real-time.
<img src="/demo/demo.gif">

## References
* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). 
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155). 
* [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629)
