import matplotlib.pyplot as plt

def plot_accuracy_per_epoch(avr_accuracy):

	plt.plot( avr_accuracy, 'r--')

	plt.ylabel('Average accuracy')

	plt.xlabel('Epochs')
	
	plt.show()