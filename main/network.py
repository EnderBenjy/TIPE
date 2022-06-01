import numpy as np
import random

results = []

class Network(object):

    def __init__(self, sizes):
        """
            size = [Ninput, Nlayer1, Nlayer2, ...]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        #https://imgur.com/R7BqAUt
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Liste matrice colonne (numpy array) avec le biais de chaque node
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Liste matrice avec les poids

    def feedforward(self, a):
        """Sortie du reseau avec a en entree"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) #Juste la formule classique, np.dot produit matriciel
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Entrainement: mini-batch stochastic gradient descent.
        Training_data: liste de (x,y) ; x entree, y sortie voulue
        Epochs: nombre d'entrainement (1 epoch = 1 fois tout le set)
        Si test_data, autoevaluation apres chaque epoch avec ce set.
        """
        training_data = list(training_data)
        n = len(training_data)
        print(epochs)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                (etape, succes)  = (j,self.evaluate(test_data))
                print(f"Epoch {etape + 1} : {succes} / {n_test}, soit {round((succes/n_test) * 100,2)}%")
            #else:
            #    print(f"Epoch {etape + 1} complete")
            #    if etape == 9:
            #       results.append(round((succes/n_test) * 100,2))

    def update_mini_batch(self, mini_batch, eta):
        """
        eta: learning rate, mini_batch: liste couple (x,y), descente sur 1 mini batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Nombre de resulats juste sur 1 evaluation (Resultat = neurone le plus active)
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
