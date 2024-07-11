import numpy as np
import os

from model import Model
from substrate import Substrate

class Population:
    def __init__(self, target, target_name, nb_indiv, nb_iter, Nb_channels, max_nb_cells, birth_rate, death_rate):
        self.target = target
        self.target_name = target_name
        self.nb_indiv = nb_indiv
        self.nb_iter = nb_iter
        self.nb_channels = Nb_channels
        self.max_nb_cells = max_nb_cells
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.pop = [Model(Nb_channels) for _ in range(nb_indiv)]

    def sort_pop(self):
        dico_eval = {indiv : indiv.evaluate(self.target, self.nb_iter, self.max_nb_cells, self.birth_rate, self.death_rate) for indiv in self.pop}
        sorted_pop = sorted(self.pop, key = lambda x : dico_eval[x])
        print( [round(float(dico_eval[indiv]), 2) for indiv in sorted_pop] )
        return sorted_pop

    def generation_step(self, T):
        sp = self.sort_pop()
        best_indiv = sp[0]
        new_pop = []
        for i in range(self.nb_indiv//3):   # keep the best third of the population and mutate them 3 times each
            new_pop.append(sp[i].mutation_perception(T))
            new_pop.append(sp[i].mutation_hidden(T))
            new_pop.append(sp[i].mutation_output(T))
        self.pop = new_pop
        return best_indiv

    def train(self, nb_generations, T, save = False):

        if save :
            path = 'train_log/'
            name = 'training_'+self.target_name+'_'
            i = 0
            while os.path.exists(path + name + str(i)):
                i += 1
            os.makedirs(path + name + str(i)) # create a new folder for the training
            path = path + name + str(i) + '/' # path to the new folder
            self.save_param(path, nb_generations, T)

        scores_log = []
        best_indiv = None
        best_score = np.inf
        for i in range(nb_generations):
            print('Generation', i+1, '/', nb_generations, ' avec T =', T)
            first_indiv = self.generation_step(T)
            first_score = first_indiv.score
            scores_log.append(first_score)
            if first_score < best_score :
                print("New best score !")
                best_score = first_indiv.score
                best_indiv = first_indiv
            print('Current best score :', best_score)
            print()

        if save :
            best_indiv.save('best_model', path)
            np.save(path + 'scores.npy', np.array(scores_log))
            
        return best_indiv, scores_log

    def save_param(self, path, nb_generations, T):
        # create a file with all the parameters of the population
        with open(path + 'parameters.txt', 'w') as f:
            f.write('target_name = ' + self.target_name + '\n')
            f.write('nb_generations = ' + str(nb_generations) + '\n')
            f.write('nb_indiv = ' + str(self.nb_indiv) + '\n')
            f.write('nb_iter = ' + str(self.nb_iter) + '\n')
            f.write('nb_channels = ' + str(self.nb_channels) + '\n')
            f.write('max_nb_cells = ' + str(self.max_nb_cells) + '\n')
            f.write('birth_rate = ' + str(self.birth_rate) + '\n')
            f.write('death_rate = ' + str(self.death_rate) + '\n')
            f.write('T = ' + str(T) + '\n')

# Each training file must contain :
    # - parameters.txt : the parameters of the training
    # - best_model/ : the best model of the training
    # - scores.npy : the scores of each generation