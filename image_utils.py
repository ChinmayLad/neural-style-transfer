import numpy as np
import torch
import cv2

class ImagePipeline:
	'''
	The tensor is trained on images in RGB pattern.
	OpenCV reads the image in BGR format. 
	'''
	def cvtTensor(self, npImage):
		npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2RGB)
		npImage = npImage.transpose(2,0,1)
		image = torch.from_numpy(npImage).type(torch.float32)
		if torch.cuda.is_available():
			image = image.cuda()
		return image

	def cvtNumpy(self, tensor):
		tensor = tensor.permute(1, 2, 0)
		image = tensor.detach().cpu().numpy()
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		image = (255*image).astype(np.uint8)
		return image


if __name__ == "__main__":

	p = ImagePipeline()

	image = np.random.randn(256,256,3)
	print("Numpy Image Shape: {}".format(image.shape))
	t = p.cvtTensor(image)
	print("Converted Tensor: {}, type: {}".format(t.shape, type(t)))
	n = p.cvtNumpy(t)
	print('Converted Numpy: {}, type: {}'.format(n.shape, type(n)))