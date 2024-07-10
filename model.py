import numpy as np
from substrate import Substrate

def sigmoid(x):
    # x is a numpy array
    # where x is above 10, the sigmoid function is approximated to 1
    # where x is below -10, the sigmoid function is approximated to 0
    return np.where(x > 10, 1, np.where(x < -10, 0, 1/(1+np.exp(-x))))
    
class Model : # the model is an update function
    def __init__(self, nb_channels) :
        self.nb_channels = nb_channels

        # FIRST LAYER
        self.weights_1 = 2*np.random.rand(3*nb_channels, 3*nb_channels)-1
        self.bias_1 = 2*np.random.rand(3*nb_channels)-1
        self.activation_1 = lambda x : sigmoid(x)

        # SECOND LAYER
        self.weights_2 = 2*np.random.rand(3*nb_channels, nb_channels)-1
        self.bias_2 = 2*np.random.rand(nb_channels)-1
        # no activation function for the second layer

        self.score = None

    def copy(self):
        new_model = Model(self.nb_channels)
        new_model.weights_1 = self.weights_1.copy()
        new_model.bias_1 = self.bias_1.copy()
        new_model.weights_2 = self.weights_2.copy()
        new_model.bias_2 = self.bias_2.copy()
        return new_model

    def update_function(self, perception_vector):
        # perception_vector : 3 * nb_channels
        # output : nb_channels
        x = np.dot(perception_vector, self.weights_1) + self.bias_1
        x = self.activation_1(x)
        x = np.dot(x, self.weights_2) + self.bias_2
        return x
    
    def mutation(self, T):  # add noise to every weights and biases
        new_model = self.copy()
        new_model.weights_1 += np.random.normal(0, T, self.weights_1.shape)
        new_model.bias_1 += np.random.normal(0, 3*self.nb_channels*T, self.bias_1.shape)
        new_model.weights_2 += np.random.normal(0, T, self.weights_2.shape)
        new_model.bias_2 += np.random.normal(0, self.nb_channels*T, self.bias_2.shape)
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
        random_channel = np.random.randint(self.nb_channels)
        for i in range(3):
            index_channel = random_channel + i*self.nb_channels
            new_model.weights_1[index_channel, :] += np.random.normal(0, T, self.weights_1.shape[1])
            new_bias = new_model.bias_1[index_channel] + np.random.normal(0, 3*self.nb_channels*T, 1)
            new_model.bias_1[index_channel] = new_bias
        return new_model
    
    def mutation_2(self, T):    # mutate the second layer
        return self.mutation_output(T).mutation_hidden(T).mutation_perception(T)
    
    def cross_over(self, other):    # average of the two models 
        new_model = Model(self.nb_channels)
        new_model.weights_1 = (self.weights_1 + other.weights_1)/2
        new_model.bias_1 = (self.bias_1 + other.bias_1)/2
        new_model.weights_2 = (self.weights_2 + other.weights_2)/2
        new_model.bias_2 = (self.bias_2 + other.bias_2)/2
        return new_model
    
    def cross_over2(self, other):   # random choice of the layers
        new_model = Model(self.nb_channels)
        if np.random.rand() > 0.5 :
            new_model.weights_1 = self.weights_1
            new_model.bias_1 = self.bias_1
            new_model.weights_2 = other.weights_2
            new_model.bias_2 = other.bias_2
        else :
            new_model.weights_1 = other.weights_1
            new_model.bias_1 = other.bias_1
            new_model.weights_2 = self.weights_2
            new_model.bias_2 = self.bias_2
        
        return new_model
    
    def evaluate(self, target, nb_steps, max_nb_cells, birth_rate, death_rate):
        if self.score == None :
            sub = Substrate(self, birth_rate, death_rate)
            i = sub.run(nb_steps, max_nb_cells)
            img, _ = sub.get_image()    # get the image of the model, values between 0 and 1
            N = img.shape[0]
            decreasing_radial_coeff = np.array([[np.sqrt(N**2//2)-np.sqrt((i-N//2)**2 + (j-N//2)**2) for j in range(N)] for i in range(N)])/np.sqrt(N**2//2)
            increasing_radial_coeff = 1 - decreasing_radial_coeff
            target_mask = target > 0  # binary mask of the target
            weighted_coeff = decreasing_radial_coeff * target_mask + 2*increasing_radial_coeff * (1-target_mask)
            square_weighted_error = np.square(np.subtract(img, target)) * weighted_coeff
            # plt.imshow(square_weighted_error, cmap = 'gray')
            # plt.show()

            if i < nb_steps :
                score = np.mean(square_weighted_error) * (nb_steps-i)
            else :
                score = np.mean(square_weighted_error) #* (1+np.log(1 + abs(30 - len(sub.pop)))) # we want to have 30 cells
            self.score = score
            return score
        return self.score