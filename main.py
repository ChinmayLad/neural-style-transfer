import cv2
import torch
from transformer import TransformNetwork
from image_utils import ImagePipeline

def main():
	path = "models/transformer-v1_2.pt"
	transformer = TransformNetwork(3, 6)
	if torch.cuda.is_available():
		transformer = transformer.cuda()
	transformer.load_model(path)

	pipeline = ImagePipeline()
	capture = cv2.VideoCapture('cat.mp4')
	counter = 0
	_, frame = capture.read()
	h, w, c = frame.shape
	four_cc = cv2.VideoWriter_fourcc('M','J','P','G')
	writer = cv2.VideoWriter('video/output3.avi', four_cc, 10, (w, h))
	# writerin = cv2.VideoWriter('video/input.avi', four_cc, 10, (w, h))
	for i in range(1000):
		_, frame = capture.read()
		if counter%10 == 0:
			image = pipeline.cvtTensor(frame)
			image = image.unsqueeze(0)
			style = transformer(image, 3)
			style = style.squeeze(0)
			out = pipeline.cvtNumpy(style)
			writer.write(out)
			cv2.imshow('Style', out)
			# writerin.write(frame)

		cv2.imshow('Original', frame)
		counter += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	writer.release
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()