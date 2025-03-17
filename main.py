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

def map_target_to_points(target, N, nb_points):
    points = []
    n = 0
    while n < nb_points :
        i = N*np.random.random()
        j = N*np.random.random()
        if target[int(j), int(i)] < 0.5 :
            points.append([j, i])
            n += 1
    return np.array(points)

        

# SUBSTRATE PARAMETERS
N = 500             # size of the substrate (and of the target image)
target_name = 'ring'    # name of the target image

# CELL PARAMETERS
nb_channels = 16     # number of channels in the model
d = 7               # equilibrium distance between cells
D = 30            # max interaction distance between cells = diameter of the cells

# CELLULAR AUTOMATON PARAMETERS
birth_rate = 0.9    # upper treshold for replication on channel 2
death_rate = -0.1    # lower treshold for death on channel 2
nb_iter = 200       # maximum number of iterations for the simulation of each model
max_nb_cells = 300  # maximum number of cells in the substrate : if there are more, the simulation stops

# EVO AlGO PARAMETERS
nb_indiv = 30       # number of individuals in the population
nb_generations = 100 # number of generations
nb_evals = nb_indiv * nb_generations # number of evaluations during the training
T = 0.01             # mutation std for the evolutionary algorithm
r_pop = 0.25        # proportion of the population taken into account for the next generation

if __name__ == '__main__':
    target_img = load_target(target_name+'.png')
    target_points = map_target_to_points(target_img, N, 100)

    plt.imshow(target_img, cmap = 'gray')
    plt.scatter(target_points[:,1], target_points[:,0], c = 'r')
    plt.show()

    pop = Population(target_img, target_points, target_name, nb_indiv, nb_iter, nb_channels, max_nb_cells, birth_rate, death_rate, N, d, D)
    print("Population created")
    best_model, scores, best_scores  = pop.train(nb_evals, T, save=True)

    X = [i for i in range(1,nb_generations+1) for _ in range(nb_indiv)]
    Y = [scores[i][j] for i in range(nb_generations) for j in range(nb_indiv)]
    plt.scatter(X, Y, alpha=0.5)
    plt.plot(best_scores, 'r')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    sub = Substrate(best_model, birth_rate, death_rate, N, d, D)
    i = sub.run(nb_iter, max_nb_cells)
    sub.display(target_img)

    sub1 = Substrate(best_model, birth_rate, death_rate, N, d, D)
    sub1.anim(nb_iter, max_nb_cells, save = True, path = '')
