import theano
from theano import tensor as T
import numpy as np
from loader.load import mnist

from plots.plot import *

from drawer.png_generator import *


def floatX(X):
	"""
	Convert to right theano type

	"""
	return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
	"""
	Initialize parameters matrix to small random values

	"""
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
	"""
	Learning model

	"""
	return T.nnet.softmax(T.dot(X, w))

trX, teX, trY, teY = mnist(onehot=True) # loading dataset





"""
Input and output as floating matrices.

Inputs are images of 28x28 = 784 pixel values.

Output is a vector of 10 values with 0 < n < 1.

"""

X = T.fmatrix()
Y = T.fmatrix()




"""
Linear system of 784 variables and 10 equations.

Parameter matrix has size 784x10

"""

w = init_weights((784, 10))





"""
py_x is a matrix of size 128x10

y_pred is a vector of size 128

"""

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)





"""
Applies cross-entropy to maximize weights of right values 

and minimize others.

Train function outputs a cost and changes weights to minimize it.

Prediction function outputs actual value.

"""

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)


avr_accuracy = [] # important!


for i in range(30):
	for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
		"""
		trX[start:end] has size 128x784
		
		trY[start:end] has size 128x10

		"""
		cost = train(trX[start:end], trY[start:end])


		"""
		Print accuracy rate for i-th epoch

		"""

	accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))

	print i, accuracy

	avr_accuracy.append(accuracy)

for i in range(10):

	filename = "digit" + str(i) + ".png"
	draw_digit(w.get_value()[:,i], filename)



plot_accuracy_per_epoch(avr_accuracy)