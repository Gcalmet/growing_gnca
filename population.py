import numpy as np
import os

from model import Model

class Population:
    def __init__(self, target_img, target_points, target_name, nb_indiv, nb_iter, Nb_channels, max_nb_cells, birth_rate, death_rate, N, d, D):
        self.target_img = target_img
        self.target_points = target_points
        self.target_name = target_name
        self.nb_indiv = nb_indiv
        self.nb_iter = nb_iter
        self.nb_channels = Nb_channels
        self.max_nb_cells = max_nb_cells
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.N = N
        self.d = d
        self.D = D
        self.pop = [Model(Nb_channels) for _ in range(nb_indiv)]

    def sort_pop(self):
        dico_eval = {indiv : indiv.evaluate(self.target_points, self.nb_iter, self.max_nb_cells, self.birth_rate, self.death_rate, self.N, self.d, self.D) for indiv in self.pop}
        sorted_pop = sorted(self.pop, key = lambda x : dico_eval[x])
        scores = [round(float(dico_eval[indiv]), 2) for indiv in sorted_pop]
        print(scores)
        return sorted_pop, scores
    
    def generation_step(self, T):   # Genetic Algorithm 
        sp, scores = self.sort_pop()
        best_indiv = sp[0]
        new_pop = []
        for i in range(self.nb_indiv):
            index_1 = min(min(np.random.randint(self.nb_indiv), np.random.randint(self.nb_indiv)), min(np.random.randint(self.nb_indiv), np.random.randint(self.nb_indiv)))  # random choice of the parents
            index_2 = min(np.random.randint(self.nb_indiv), np.random.randint(self.nb_indiv))
            while index_1 == index_2 :
                index_2 = min(np.random.randint(self.nb_indiv), np.random.randint(self.nb_indiv))
            ind_1 = sp[index_1]
            ind_2 = sp[index_2]
            if np.random.random() < 0.25 :
                child = ind_1.cross_over2(ind_2).mutation(T)    # cross over and mutation
            else :
                child = ind_1.mutation(T)                        # mutation
            score = child.evaluate(self.target_points, self.nb_iter, self.max_nb_cells, self.birth_rate, self.death_rate, self.N, self.d, self.D)
            child.score = score
            new_pop.append(child)
        
        self.pop = new_pop
        return best_indiv, scores

    def train(self, nb_evals, T, save = False): 

        scores_log = []

        nb_gen = nb_evals // self.nb_indiv

        if save :
            path = 'train_log/'
            name = 'training_'+self.target_name+'_'
            i = 0
            while os.path.exists(path + name + str(i)):
                i += 1
            os.makedirs(path + name + str(i)) # create a new folder for the training
            path = path + name + str(i) + '/' # path to the new folder
            self.save_param(path, nb_evals, T)

        best_scores_log = []
        best_indiv = None
        best_score = np.inf
        for i in range(nb_gen):
            print('Generation', i+1, '/', nb_gen, ' avec T =', T)
            first_indiv, scores = self.generation_step(T)
            scores_log.append(scores)
            first_score = first_indiv.score
            best_scores_log.append(first_score)
            if first_score < best_score :
                print("New best score !")
                best_score = first_indiv.score
                best_indiv = first_indiv
            print('Current best score :', best_score)
            print()

        if save :
            best_indiv.save('best_model', path)
            np.save(path + 'scores.npy', np.array(scores_log))
            
        return best_indiv, scores_log, best_scores_log

    def save_param(self, path, nb_evals, T):
        # create a file with all the parameters of the population
        with open(path + 'parameters.txt', 'w') as f:
            f.write('target_name = ' + self.target_name + '\n')
            f.write('nb_evals = ' + str(nb_evals) + '\n')
            f.write('nb_indiv = ' + str(self.nb_indiv) + '\n')
            f.write('nb_iter = ' + str(self.nb_iter) + '\n')
            f.write('nb_channels = ' + str(self.nb_channels) + '\n')
            f.write('max_nb_cells = ' + str(self.max_nb_cells) + '\n')
            f.write('birth_rate = ' + str(self.birth_rate) + '\n')
            f.write('death_rate = ' + str(self.death_rate) + '\n')
            f.write('N = ' + str(self.N) + '\n')
            f.write('d = ' + str(self.d) + '\n')
            f.write('D = ' + str(self.D) + '\n')
            f.write('T = ' + str(T) + '\n')
            f.write('r_pop = ' + str(0.25) + '\n')

# Each training file must contain :
    # - parameters.txt : the parameters of the training
    # - best_model/ : the best model of the training
    # - scores.npy : the scores of each generation