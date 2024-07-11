import numpy as np
import matplotlib.pyplot as plt

from population import Population
from substrate import Substrate
from model import Model

def load_target(name):  # suposing the image is N x N and png format
    target = plt.imread('targets/'+name)
    if len(target.shape) == 3:
        target = target[:,:,0]  # keep only the first channel
    if target[0,0] <= 1:
        target = np.array(target*255, dtype = np.uint8) # convert to uint8 (0-255)
    return target

def load_model(name, path):   # load a model from the folder path+name/   (ex path = 'train_log/training_ring_0/', name = 'best_model')
    weights_1 = np.load(path + name + '/weights_1.npy', allow_pickle = True)
    bias_1 = np.load(path + name + '/bias_1.npy', allow_pickle = True)
    weights_2 = np.load(path + name + '/weights_2.npy', allow_pickle = True)
    bias_2 = np.load(path + name + '/bias_2.npy', allow_pickle = True)
    score = np.load(path + name + '/score.npy', allow_pickle = True)
    nb_channels = bias_2.shape[0]
    model = Model(nb_channels, random = False, weights_1 = weights_1, bias_1 = bias_1, weights_2 = weights_2, bias_2 = bias_2)
    model.score = score
    return model

# SUBSTRATE PARAMETERS
N = 500             # size of the substrate (and of the target image)
target_name = 'circle'    # name of the target image

# CELL PARAMETERS
nb_channels = 5     # number of channels in the model
d = 7               # equilibrium distance between cells
D = 30              # max interaction distance between cells + diameter of the cells

# CELLULAR AUTOMATON PARAMETERS
birth_rate = 0.9    # upper treshold for replication on channel 2
death_rate = 0.2    # lower treshold for death on channel 0
nb_iter = 200       # number of iterations for the simulation of each model
max_nb_cells = 500  # maximum number of cells in the substrate : if there are more, the simulation stops

# GENETIC ALGORITHM PARAMETERS
nb_indiv = 15       # number of individuals in the population
nb_generations = 10 # number of generations for the genetic algorithm
T = 0.2             # mutation std for the genetic algorithm

if __name__ == '__main__':
    target = load_target(target_name+'.png')

    # pop = Population(target, target_name, nb_indiv, nb_iter, nb_channels, max_nb_cells, birth_rate, death_rate)
    # best_model, scores = pop.train(nb_generations, T, save = True)

    # plt.plot(scores)
    # plt.show()

    # sub = Substrate(best_model, birth_rate, death_rate)
    # i = sub.run(nb_iter, max_nb_cells)
    # sub.display(target)

    # sub1 = Substrate(best_model, birth_rate, death_rate)
    # sub1.anim(nb_iter, max_nb_cells, save = False)

    model = load_model('best_model', 'train_log/training_circle_0/')
    sub = Substrate(model, birth_rate, death_rate)
    i = sub.run(nb_iter, max_nb_cells)
    sub.display(target)

    sub1 = Substrate(model, birth_rate, death_rate)
    sub1.anim(nb_iter, max_nb_cells, save = False)
