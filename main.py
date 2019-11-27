import cv2
import torch
import numpy as np
from transformer import TransformNetwork
from image_utils import ImagePipeline

def main():
	# load model
	path = "models/transformer-v1_2.pt"
	transformer = TransformNetwork(3, 6)
	if torch.cuda.is_available():
		transformer = transformer.cuda()
	transformer.load_model(path)

	# create Image pipeline for handling tensors
	pipeline = ImagePipeline()

	# initialize video capture
	capture = cv2.VideoCapture('cat.mp4')
	counter = 0
	_, frame = capture.read()
	h, w, c = frame.shape
	# four_cc = cv2.VideoWriter_fourcc('M','J','P','G')
	# writer = cv2.VideoWriter('video/output10.avi', four_cc, 10, (w+w-200, h))
	# writerin = cv2.VideoWriter('video/input.avi', four_cc, 10, (w, h))
	while capture.isOpened():
		ret, frame = capture.read()
		if ret:
			if counter%10 == 0:
				image = pipeline.cvtTensor(frame)
				image = image.unsqueeze(0)
				style = transformer(image, 5)
				style = style.squeeze(0)
				out = pipeline.cvtNumpy(style)
				out1 = cv2.hconcat([frame,out])
				# writer.write(out1)
				cv2.imshow('demo', out1)
				# writerin.write(frame)
			# cv2.imshow('Original', frame)
			counter += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	capture.release()
	# writer.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()