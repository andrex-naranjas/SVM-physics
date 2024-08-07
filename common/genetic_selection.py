"""
---------------------------------------------------
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
---------------------------------------------------
"""
import numpy as np
import random
from random import randint
from tqdm import tqdm
from functools import lru_cache
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,auc,precision_score,roc_auc_score,f1_score,recall_score
# framework includes
from common.common_methods import roc_curve_adaboost
from common.svm_methods import compute_kernel


class GeneticSelection:
    """
    Genetic algorithm for training sub-dataset selection
    It returns an array that contain the selected indexes
    """
    def __init__(self, model, model_type, is_precom, kernel_fcn, X_train, Y_train, X_test, Y_test, pop_size, chrom_len, n_gen, coef, mut_rate, score_type='acc', selec_type='tournament'):
        self.model = model
        self.model_type = model_type
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test
        if len(self.Y_test) > len(self.Y_train):
            self.X_test  = resample(X_test, replace=False, n_samples=10000)
            self.Y_test  = resample(Y_test, replace=False, n_samples=10000)
        self.y0_index = Y_train[Y_train == 0].index
        self.y1_index = Y_train[Y_train == 1].index
        self.population_size=pop_size
        self.chrom_len = chrom_len
        self.n_generations = n_gen # maximum number of iterations
        self.coef = coef
        self.mutation_rate = mut_rate
        self.score_type = score_type
        self.selec_type = selec_type
        if(model_type == 'absv'):
            self.AB_SVM = True
        else:
            self.AB_SVM = False
        self.is_precom = is_precom=="precomp"
        self.kernel_fcn = kernel_fcn                    

    def execute(self):
        """
        Method to execute the genetic selections
        """
        best_chromo = np.array([])
        best_score  = []
        next_generation_x, next_generation_y, next_generation_indexes = self.initialize_population(self.population_size, self.chrom_len)
        for generation in tqdm(range(self.n_generations)):
            n_p_y = len(next_generation_y[next_generation_y==1])
            n_n_y = len(next_generation_y[next_generation_y==0])
            print(n_p_y, n_n_y, n_p_y + n_n_y, 'balance check for every generation')
            scores, popx, popy, index = self.fitness_score(next_generation_x, next_generation_y, next_generation_indexes)
            scores, popx, popy, index = self.set_population_size(scores, popx, popy, index, generation, self.population_size)
            if self.termination_criterion(generation, best_score, window=10):
                print('End of genetic algorithm')
                break
            else:
                if(self.selec_type == 'roulette'):
                    pa_x, pa_y, pb_x, pb_y, ind_a, ind_b = self.roulette_wheel(scores, popx, popy, index)
                elif(self.selec_type == 'tournament'):
                    pa_x, pa_y, pb_x, pb_y, ind_a, ind_b = self.tournament_selection(scores, popx, popy, index)
                else:
                    pa_x, pa_y, pb_x, pb_y, ind_a, ind_b = self.selection(popx, popy, index, self.coef)
                new_population_x , new_population_y, new_index = self.crossover(pa_x, pa_y, pb_x, pb_y, ind_a, ind_b, self.population_size)
                new_offspring_x, new_offspring_y, new_offs_index = self.mutation(new_population_x, new_population_y, new_index, self.mutation_rate)
                best_score.append(scores[0])
                next_generation_x, next_generation_y, next_generation_indexes = self.append_offspring(popx, popy, new_offspring_x, new_offspring_y,
                                                                                                      index, new_offs_index)
            print(f"Best score achieved in generation {generation} is {scores[0]}")
        self.best_pop = index

    def best_population(self):
        """
        Fetches the best trained indexes, removing repetitions
        """
        best_train_indexes = np.unique(self.best_pop.flatten())
        return np.unique(best_train_indexes)

    def initialize_population(self, size, chromosome_length):
        """
        Initalize the population with size==pop_size
        """
        population_x, population_y, index_pop = [], [], []
        for i in range(size):
            chromosome_x, chromosome_y = self.get_subset(size=chromosome_length, count=i)
            population_x.append(chromosome_x.values)
            population_y.append(chromosome_y.values)
            # keep track of the indexes and propagate them during the GA selection
            index_pop.append(chromosome_x.index)
        return np.array(population_x), np.array(population_y), np.array(index_pop)

    def get_subset(self, size, count): #size==chrom_size
        """
        Construct chromosomes
        """
        # Set the size, to prevent larger sizes than allowed
        if(len(self.y0_index) < size/2 or len(self.y1_index) < size/2):
            size = np.amin([len(self.y0_index), len(self.y1_index)])

        # Select a random subset of indexes of length size/2
        random_y0 = np.random.choice(self.y0_index, int(size/2), replace = False)
        random_y1 = np.random.choice(self.y1_index, int(size/2), replace = False)

        # Concatenate indexes for balanced dataframes
        indexes = np.concatenate([random_y0, random_y1])

        # Construct balanced datasets
        X_balanced = self.X_train.loc[indexes]
        y_balanced = self.Y_train.loc[indexes]

        # Shuffled dataframes
        rand_st = randint(0, 10)
        X_balanced = X_balanced.sample(frac=1, random_state=rand_st)
        y_balanced = y_balanced.sample(frac=1, random_state=rand_st)

        # It may exist repeated indexes that change the chromosome size
        # These lines fix the issue coming from bootstrap
        # The GA selection cannot handle different chromosome sizes
        if(len(X_balanced) != len(indexes)):
            X_balanced = resample(X_balanced, replace=False, n_samples=len(indexes))
            y_balanced = resample(y_balanced, replace=False, n_samples=len(indexes))

        return X_balanced, y_balanced

    @lru_cache(maxsize = 1000)
    def memoization_score(self, tuple_chrom_x , tuple_chrom_y):
        """
        Helper method to better handle the cache memory
        """
        chromosome_x, chromosome_y = np.asarray(tuple_chrom_x), np.asarray(tuple_chrom_y)
        if self.is_precom: # pre-compute the kernel matrices if requested
            matrix_train = compute_kernel(self.kernel_fcn, chromosome_x)
            X_test = compute_kernel(self.kernel_fcn, chromosome_x, self.X_test)
            self.model.fit(matrix_train, chromosome_y[0])
        else:
            X_test = self.X_test
            self.model.fit(chromosome_x, chromosome_y[0])
        if(self.AB_SVM and self.model.n_classifiers==0): return 0.
        predictions = self.model_predictions(X_test, self.model_type, self.score_type)
        score       = self.score_value(self.Y_test, predictions, self.model_type, self.score_type)
        return score

    def fitness_score(self, pop_x, pop_y, indexes_pop):
        """
        Method to obtain the metrics that define a certaun chromosome
        """
        scores = np.array([])
        for chromosome_x, chromosome_y in zip(pop_x, pop_y):
            array_tuple_x = map(tuple, chromosome_x)
            array_tuple_y = map(tuple, chromosome_y.reshape((1, len(chromosome_y))))
            tuple_tuple_x = tuple(array_tuple_x)
            tuple_tuple_y = tuple(array_tuple_y)
            score         = self.memoization_score(tuple_tuple_x , tuple_tuple_y)
            scores        = np.append(scores, score)
            if self.AB_SVM:  self.model.clean() # needed for AdaBoostSVM
        # Indexes sorted by score, see the cross check!
        sorted_indexes  = np.argsort(-1*scores)
        return scores[sorted_indexes], pop_x[sorted_indexes], pop_y[sorted_indexes], indexes_pop[sorted_indexes]

    def set_population_size(self, scores, popx, popy, index, generation, size):
        """
        Gets rid of lower part of population, restoring original size
        """
        if generation == 0:
            pass
        else:
            scores = scores[:size]
            popx   = popx[:size]
            popy   = popy[:size]
            index  = index[:size]
        return scores, popx, popy, index

    def selection(self, pop_x, pop_y, data_index, coef=0.5):
        """
        High-Low-fit selection
        high fit and low fit parts of population
        """
        indices = np.array([i for i in range(len(pop_x))])
        hf_indexes = indices[:int(len(indices)*coef)]
        lf_indexes = indices[int(len(indices)*coef):]

        hf = np.random.choice(hf_indexes, 1, replace=False)
        lf = np.random.choice(lf_indexes, 1, replace=False)

        pa_x = pop_x[hf]
        pa_y = pop_y[hf]
        in_a = data_index[hf]

        pb_x = pop_x[lf]
        pb_y = pop_y[lf]
        in_b = data_index[lf]

        return pa_x, pa_y, pb_x, pb_y, in_a, in_b

    def tournament_selection(self, scores, popx, popy, data_index, size_k=3):
        """
        Tournament selection
        """
        indices = np.array([i for i in range(len(popx))])
        parents_indices = []

        for _ in range(2):
            competitors = np.random.choice(indices, size_k, replace=False)
            best_competitor = np.min(competitors)
            indices = np.delete(indices, best_competitor)
            parents_indices.append(best_competitor)

        pa_x = [popx[parents_indices[0]]]
        pa_y = [popy[parents_indices[0]]]
        in_a = [data_index[parents_indices[0]]]

        pb_x = [popx[parents_indices[1]]]
        pb_y = [popy[parents_indices[1]]]
        in_b = [data_index[parents_indices[1]]]

        return pa_x, pa_y, pb_x, pb_y, in_a, in_b

    def roulette_wheel(self, scores, popx, popy, data_index):
        """
        Roulette wheel selection
        """
        chosen = []
        # Making index-fitness dictionary
        local_indexes = np.array([i for i in range(len(scores))])
        zip_iterator = zip(local_indexes, scores)
        index_fitness_dictionary = dict(zip_iterator)

        # Compute probabilities
        fitness = index_fitness_dictionary.values()
        total_fit = float(sum(fitness))
        relative_fitness = [f/total_fit for f in fitness]
        cumulative_probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

        # Select two local indexes
        # If r < qi select the first chromosome else Select the individual xi such that qi-1 < r < qi
        for n in range(2):
            r = random.random()
            for i in local_indexes:
                if r <= cumulative_probabilities[i]:
                    chosen.append(i)
                    break

        # Retrieve parents and global indexes with local indexes
        pa_x = [popx[chosen[0]]]
        pa_y = [popy[chosen[0]]]
        in_a = [data_index[chosen[0]]]

        pb_x = [popx[chosen[1]]]
        pb_y = [popy[chosen[1]]]
        in_b = [data_index[chosen[1]]]

        return pa_x, pa_y, pb_x, pb_y, in_a, in_b

    def crossover(self, parent_a_x, parent_a_y, parent_b_x, parent_b_y, index_a, index_b, num_children):
        """
        Perform the chromosomes crossover
        """
        offspring_x = []
        offspring_y = []
        offspring_index = []

        for i in range(0, num_children):
            p_ab_x = np.array([])
            p_ab_y = np.array([])
            i_ab = np.array([])

            # Generate random indices
            rand_indx = np.random.choice(range(0,2*len(parent_a_x[0])), len(parent_a_x[0]), replace=False)

            p_ab_x = np.concatenate((parent_a_x[0], parent_b_x[0]), axis=0)
            p_ab_y = np.concatenate((parent_a_y[0], parent_b_y[0]), axis=0)
            i_ab = np.concatenate((index_a[0], index_b[0]), axis=0)

            new_x = p_ab_x[rand_indx]
            new_y = p_ab_y[rand_indx]
            new_i = i_ab[rand_indx]

            offspring_x.append(new_x)
            offspring_y.append(new_y)
            offspring_index.append(new_i)

        return np.array(offspring_x), np.array(offspring_y), np.array(offspring_index)

    def mutation(self, offspring_x, offspring_y, index, mutation_rate):
        """
        Mutation method
        """
        pop_nextgen_x = []
        pop_nextgen_y = []
        ind_nextgen = []

        for i in range(0, len(offspring_x)):
            chromosome_x = offspring_x[i]
            chromosome_y = offspring_y[i]
            index_chromosome = index[i]

            for j in range(len(chromosome_x)):
                if random.random() < mutation_rate:
                    while True:
                        # Check balance of chromosome
                        n_p_y = len(chromosome_y[chromosome_y == 1])
                        # If already balanced
                        if n_p_y == len(chromosome_y)/2:
                            if chromosome_y[j] == 1:
                                # Get random sample from X_train, Y_train=1
                                random_x, random_y, random_index = self.get_random_gene(self.y1_index)
                            else:
                                # Get random sample from X_train, Y_train=-1
                                random_x, random_y, random_index = self.get_random_gene(self.y0_index)
                        # If class 1 is outnumbered (balance dataset)
                        elif n_p_y < len(chromosome_y)/2:
                            random_x, random_y, random_index = self.get_random_gene(self.y1_index)
                        # If class -1 is outnumbered (balance dataset)
                        elif n_p_y > len(chromosome_y)/2:
                            random_x, random_y, random_index = self.get_random_gene(self.y0_index)

                        # Check if new random gene is already in the population. If not, it is added
                        if (chromosome_x == random_x.to_numpy()).all(1).any() is not True:
                            chromosome_x[j] = random_x.to_numpy()
                            chromosome_y[j] = int(random_y.to_numpy()) # partially fix a bug for ecoli,car,cancer(updated pip packs)
                            index_chromosome[j] = random_index.to_numpy()
                            break

            pop_nextgen_x.append(chromosome_x)
            pop_nextgen_y.append(chromosome_y)
            ind_nextgen.append(index_chromosome)

        return np.array(pop_nextgen_x), np.array(pop_nextgen_y), np.array(ind_nextgen) # check existence of genes -1 and 1 in Y, to avoid sklearn crashes


    def get_random_gene(self, class_type_index):
        """
        Method to get random sample from X_train, Y_train
        function (used to balance data in mutation function)
        """
        rand_st  = randint(0, 10)
        random_x = self.X_train.loc[class_type_index].sample(random_state=rand_st)
        random_y = self.Y_train.loc[class_type_index].sample(random_state=rand_st)
        random_index = random_x.index
        return random_x, random_y, random_index

    def append_offspring(self, next_generation_x, next_generation_y, new_offspring_x, new_offspring_y, next_generation_indexes, new_off_index):
        """
        Append offspring to population
        """
        next_generation_x = np.append(next_generation_x, new_offspring_x, axis=0)
        next_generation_y = np.append(next_generation_y, new_offspring_y, axis=0)
        next_generation_indexes = np.append(next_generation_indexes, new_off_index, axis=0)
        return  next_generation_x, next_generation_y, next_generation_indexes


    def termination_criterion(self, generation, best_score, window=10):
        """
        Method that defines the termination criterion returning a boolen
        """
        if generation <= window - 1:
            return False
        else:
            std = pd.Series(best_score).rolling(window).std() # equivalent to np.std(best_score, ddof=1)
            if std.iloc[len(std)-1] < 0.01:
                return True
            else:
                return False

    def score_value(self, Y_test, y_pred, model_type, score_type):
        """
        Method to compute different scores given options
        """
        Y_test = Y_test.astype(float).values # make Y_test and y_pred same type
        if(score_type == 'auc' and model_type == 'absv'):
            TPR, FPR = roc_curve_adaboost(y_pred, Y_test)
            score_value = auc(FPR,TPR)
        elif(score_type == 'auc' and model_type != 'absv'):
            score_value = roc_auc_score(Y_test, y_pred)
        elif(score_type == 'acc'):
            score_value = accuracy_score(Y_test, y_pred)
        elif(score_type == 'prec'):
            score_value = precision_score(Y_test, y_pred)
        elif(score_type == 'f1'):
            score_value = f1_score(Y_test, y_pred)
        elif(score_type == 'rec'):
            score_value = recall_score(Y_test, y_pred)
        elif(score_type == 'gmean'):
            score_value  = np.sqrt(precision_score(Y_test, y_pred)*recall_score(Y_test, y_pred))
        return score_value

    def model_predictions(self, X_test, model_type, score_type):
        """
        Method computes the prediction given the score type set
        """
        X_test = self._check_X(X_test)
        if(score_type == 'auc'):
            if(model_type == 'absv'):
                return self.model.decision_thresholds(X_test, glob_dec=True)
            elif(model_type == 'prob'):
                return self.model.predict_proba(X_test)[:,1]
            elif(model_type == 'deci'):
                return self.model.predict(X_test) # self.model.decision_function(X_test)
        else:
            return self.model.predict(X_test)

    def _check_X(self, X):
        """
        Validate assumptions about format of input data. np.array is expected
        """
        if type(X) == type(np.array([])):
            return X
        else:
            # Convert pandas into numpy arrays
            X = X.values
            return X
