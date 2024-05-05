# import the necessary packages
from tensorflow.keras.datasets import mnist
import numpy as np

def load_az_dataset(datasetPath):
	
	data = []
	labels = []
	
	for row in open(datasetPath):
		# parse the label and image 
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))

		data.append(image)
		labels.append(label)
		
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")

	return (data, labels)

def load_mnist_dataset():
	# load the MNIST dataset and stack the training data and testing
	# data together (we'll create our own training and testing splits
	# later in the project)
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	# return a 2-tuple of the MNIST data and labels
	return (data, labels)