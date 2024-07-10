import numpy as np

from model import Model
from substrate import Substrate

class Population:
    def __init__(self, target, nb_indiv, nb_iter, Nb_channels, max_nb_cells, birth_rate, death_rate):
        self.target = target
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
        for i in range(self.nb_indiv//3):
            new_pop.append(sp[i].mutation_perception(T))
            new_pop.append(sp[i].mutation_hidden(T))
            new_pop.append(sp[i].mutation_output(T))
            #new_pop.append(sp[i].cross_over2(sp[i+1]))

        self.pop = new_pop
        return best_indiv

    def train(self, nb_generations, T = 0.2):
        best_indiv = None
        best_score = np.inf
        for i in range(nb_generations):
            print('Generation', i+1, '/', nb_generations, ' avec T =', T)
            first_indiv = self.generation_step(T)
            if first_indiv.score < best_score :
                print("New best score !")
                best_score = first_indiv.score
                best_indiv = first_indiv
                """sub = Substrate(best_indiv)
                print(nb_iter)
                sub.run(nb_iter, self.max_nb_cells)
                img, _ = sub.get_image()
                plt.imshow(img, cmap = 'gray')
                plt.show()"""
            print('Current best score :', best_score)
            print()
        return best_indiv
    
    def exibit(self, model, save_anim = False):
        sub = Substrate(model, self.birth_rate, self.death_rate)
        i = sub.run(self.nb_iter, self.max_nb_cells)
        print('Simulation stopped at iteration', i)
        sub.display(self.target)

        sub1 = Substrate(model, self.birth_rate, self.death_rate)
        sub1.anim(self.nb_iter, save = save_anim)