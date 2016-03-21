import png
import numpy as np
import os

opacity_rate = 100 # recommended

def draw_digit(weights, filename):

	directory = "./generated/"

	filepath = os.path.join(directory, filename)

	f = open(filepath, 'wb')      # binary mode is important
	w = png.Writer(28, 28, greyscale=True)
	for i in range(len(weights)):

		weights[i] = ((weights[i] - min(weights))/(max(weights) - min(weights))) * 255 * opacity_rate

		if weights[i] < 0:

			weights[i] = 0

		if weights[i] > 255:

			weights[i] = 255



	weights = weights.reshape((28,28))
	w.write(f, weights.tolist())
	f.close()

