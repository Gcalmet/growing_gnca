import numpy as np
import os
from substrate import Substrate
import ot

def sigmoid(x):
    # x is a numpy array
    # where x is above 10, the sigmoid function is approximated to 1
    # where x is below -10, the sigmoid function is approximated to 0
    return np.where(x > 10, 1, np.where(x < -10, -1, 2/(1+np.exp(-x)) - 1))


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
class Model : # the model is an update function
    def __init__(self, nb_channels, random = True, weights_1 = None, bias_1 = None, weights_2 = None, bias_2 = None):
        self.nb_channels = nb_channels

        if not random :  # we suppose that the weights and biases are not None

            # FIRST LAYER
            self.weights_1 = weights_1
            self.bias_1 = bias_1
            self.activation_1 = lambda x : np.maximum(0, x)

            # SECOND LAYER
            self.weights_2 = weights_2
            self.bias_2 = bias_2
            # no activation function for the second layer


        else :
            # FIRST LAYER
            self.weights_1 = 2*np.random.rand(3*nb_channels, 128)-1
            self.bias_1 = 2*np.random.rand(128)-1
            self.activation_1 = lambda x : np.maximum(0, x)

            # SECOND LAYER
            self.weights_2 = 2*np.random.rand(128, nb_channels)-1
            self.bias_2 = 2*np.random.rand(nb_channels)-1
            # no activation function for the second layer

        self.score = None

    def copy(self):
        new_weights_1 = self.weights_1.copy()
        new_bias_1 = self.bias_1.copy()
        new_weights_2 = self.weights_2.copy()
        new_bias_2 = self.bias_2.copy()
        new_model = Model(self.nb_channels, random = False, weights_1 = new_weights_1, bias_1 = new_bias_1, weights_2 = new_weights_2, bias_2 = new_bias_2)
        return new_model

    def forward_pass(self, values, sobel_x, sobel_y): # values : 3 * nb_channels, sobel_x : nb_channels, sobel_y : nb_channels
        # perception_vector : 3 * nb_channels
        # output : nb_channels
        x = np.concatenate((values, sobel_x, sobel_y))
        x = np.dot(x, self.weights_1) + self.bias_1
        x = self.activation_1(x)
        x = np.dot(x, self.weights_2) + self.bias_2
        return x
    
    def mutation(self, T):  # add noise to every weights and biases
        new_model = self.copy()
        if np.random.rand() < 0.25 :
            new_model.weights_1 += np.random.normal(0, T, self.weights_1.shape)

        if np.random.rand() < 0.25 :
            new_model.bias_1 += np.random.normal(0, T, self.bias_1.shape)

        if np.random.rand() < 0.25 :
            new_model.weights_2 += np.random.normal(0, T, self.weights_2.shape)

        if np.random.rand() < 0.25 :
            new_model.bias_2 += np.random.normal(0, T, self.bias_2.shape)

        return new_model
    
    def mutation_perception(self, T): # mutate how one channel is perceived (Sobelx, Sobely, I) (weights_1)
        new_model = self.copy()
        random_channel = np.random.randint(self.nb_channels)
        for i in range(3):
            index_channel = random_channel + i*self.nb_channels
            new_model.weights_1[:, index_channel] += np.random.normal(0, T, self.weights_1.shape[0])
        return new_model
    
    def mutation_output(self, T):    # mutate how one channel is outputed
        new_model = self.copy()
        random_channel = np.random.randint(self.nb_channels)
        new_model.weights_2[random_channel, :] += np.random.normal(0, T, self.weights_2.shape[1])
        new_bias = new_model.bias_2[random_channel] + np.random.normal(0, self.nb_channels*T, 1)
        new_model.bias_2[random_channel] = new_bias
        return new_model
    
    def mutation_hidden(self, T):    # mutate how one channel is interpreted in the hidden layer
        new_model = self.copy()
        random_channel = np.random.randint(3*self.nb_channels)
        new_model.weights_1[random_channel, :] += np.random.normal(0, T, self.weights_1.shape[1])
        new_model.bias_1[random_channel] += np.random.normal(0, T, 1)
        return new_model
    
    def mutation_2(self, T):    # mutate every layer
        return self.mutation_output(T).mutation_hidden(T).mutation_perception(T)
    
    def cross_over(self, other):    # average of the two models 
        new_model_weights_1 = (self.weights_1 + other.weights_1)/2
        new_model_bias_1 = (self.bias_1 + other.bias_1)/2
        new_model_weights_2 = (self.weights_2 + other.weights_2)/2
        new_model_bias_2 = (self.bias_2 + other.bias_2)/2
        new_model = Model(self.nb_channels, random = False, weights_1 = new_model_weights_1, bias_1 = new_model_bias_1, weights_2 = new_model_weights_2, bias_2 = new_model_bias_2)
        return new_model
    
    def cross_over2(self, other):   # random choice of the layers
        if np.random.rand() > 0.5 :
            new_model_weights_1 = self.weights_1
            new_model_bias_1 = self.bias_1
            new_model_weights_2 = other.weights_2
            new_model_bias_2 = other.bias_2
        else :
            new_model_weights_1 = other.weights_1
            new_model_bias_1 = other.bias_1
            new_model_weights_2 = self.weights_2
            new_model_bias_2 = self.bias_2
        new_model = Model(self.nb_channels, random = False, weights_1 = new_model_weights_1, bias_1 = new_model_bias_1, weights_2 = new_model_weights_2, bias_2 = new_model_bias_2)
        return new_model

    def evaluate(self, target_points, nb_steps, max_nb_cells, birth_rate, death_rate, N, d, D): # target is an array of points
        if self.score == None :
            sub = Substrate(self, birth_rate, death_rate, N, d, D)
            i = sub.run(nb_steps, max_nb_cells)
            input = np.array([[cell.x, cell.y, cell.values[0]] for cell in sub.pop if cell.values[0] > 0])
            nb_positions = len(input)
            if nb_positions == 0 :
                self.score = np.inf
                return np.inf
            nb_target_points = len(target_points)
            input_positions = input[:,0:2]
            input_weights = input[:,2]
            C = ot.dist(input_positions, target_points)
            a = input_weights/sum(input_weights)
            b = np.ones(shape=nb_target_points)/nb_target_points
            W = ot.emd(a, b, C)   #a, b are the weights of the distributions
            # the output of ot.emd is a matrix of size len(a) * len(b) where W[i,j] is the amount of mass that goes from a[i] to b[j]
            # we want to compute the wasserstein distance between the two distributions
            self.score = np.sum(W*C)
        return self.score
    
    def save(self, name, path): # save the model in the folder path+name/ and create 5 files in it : weights_1, bias_1, weights_2, bias_2, score
        path = path + name + '/'
        # create a new folder for the model if it does not exist
        i = 0
        while os.path.exists(path + str(i)):
            i += 1
        os.makedirs(path + str(i))
        np.save(path + 'weights_1', self.weights_1, allow_pickle = True)
        np.save(path + 'bias_1', self.bias_1, allow_pickle = True)
        np.save(path + 'weights_2', self.weights_2, allow_pickle = True)
        np.save(path + 'bias_2', self.bias_2, allow_pickle = True)
        np.save(path + 'score', self.score, allow_pickle = True)
        print('Model saved')

def load(name, path):   # load a model from the folder path+name/   (ex path = 'train_log/training_ring_0/', name = 'best_model')
    weights_1 = np.load(path + name + '/weights_1.npy', allow_pickle = True)
    bias_1 = np.load(path + name + '/bias_1.npy', allow_pickle = True)
    weights_2 = np.load(path + name + '/weights_2.npy', allow_pickle = True)
    bias_2 = np.load(path + name + '/bias_2.npy', allow_pickle = True)
    score = np.load(path + name + '/score.npy', allow_pickle = True)
    nb_channels = bias_2.shape[0]
    model = Model(nb_channels, random = False, weights_1 = weights_1, bias_1 = bias_1, weights_2 = weights_2, bias_2 = bias_2)
    model.score = score
    return model

def clear_test_folder():
    # remove every folder in the test folder
    path = 'train_log/test/'
    # remove the folder test by force
    os.system('rm -rf ' + path)
    os.makedirs(path)
    print('Test folder cleared')


if __name__ == '__main__':
    # test of saving and loading a model
    new_random_model = Model(5)
    print(new_random_model.weights_1)
    clear_test_folder()
    new_random_model.save('model', 'train_log/test/')
    new_model = load('model', 'train_log/test/')
    print(new_model.weights_1)
