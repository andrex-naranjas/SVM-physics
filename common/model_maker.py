"""
---------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales
---------------------------------------------------
"""
import numpy as np

# support vector machines
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# framework includes
from common.svm_methods import LaplacePrecomputed
from common.svm_methods import PolyPrecomputed
from common.svm_methods import KernelSum
from common.svm_methods import KernelProd
from common.boosted_svm import BoostedSVM

# n-word classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# discriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# probabilistic
from sklearn.naive_bayes      import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

# linear models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


def custom_svm(my_c=100, my_gamma_end=100, myKernel="rbf", myDegree=1, myCoef0=1):
  
    return SVC(C=my_c, kernel=myKernel, gamma=my_gamma_end, degree=myDegree, coef0=myCoef0,
               shrinking = True, probability = False, tol = 0.001) # probability = True


def adaboost_svm(div_flag=False, my_c=150, my_gamma_end=100, myKernel='rbf', myDegree=1, myCoef0=1, early_stop=False, debug=True):
    # boosted support vector machine (ensemble)
    svmb = BoostedSVM(C=my_c, gammaEnd=my_gamma_end, myKernel=myKernel, myDegree=myDegree, myCoef0=myCoef0,
                      Diversity=div_flag, early_stop=early_stop, debug=debug)
    return svmb

def single_svm(my_kernel):
    # support vector machine (single case)
    my_C = 100
    my_gamma = 100
    my_coef = +1
    if my_kernel == 'sigmoid':
        my_coef = -1
        my_gamma = 10
    elif my_kernel == 'poly':
        my_C = 10
        my_gamma = 0.1            
    return SVC(kernel=my_kernel, shrinking = True, probability = False, tol = 0.001)#, degree=2, coef0=my_coef, gamma=my_gamma, shrinking = True, probability = True, tol = 0.001)


def linear_svm():
    # support vector machine (linear case)
    # decision
    return LinearSVC(C=10.0)
    
def neural_net():
    # Neural Network Multi Layer Perceptron classifier
    return MLPClassifier(solver='sgd', random_state=1)

def k_neighbors():
    # K neighbors classifier. n_neighbors=3 because there are 2 classes
    return KNeighborsClassifier(n_neighbors=3)

def linear_dis():
    # to-do set values
    return LinearDiscriminantAnalysis()

def quad_dis():
    # to-do set values
    return QuadraticDiscriminantAnalysis()

def gauss_nb():
    # to-do: set values    
    return GaussianNB()

def gauss_pc():
    # to-do: set values
    return GaussianProcessClassifier()

def log_reg():
    # to-do: set values
    return LogisticRegression()

def ridge_class():
    # to-do: set values
    # decision
    return RidgeClassifier()

def sgdc_class():
    # to-do: set values
    # decision
    return SGDClassifier()

def pass_agre():
    # to-do: set values
    # decision
    return PassiveAggressiveClassifier()

def model_loader_batch(process, exotic_single='exotic'):
    # return a single model to be used in a batch job
    if exotic_single =='exotic':
        batch_models = model_flavors_exotic()
    elif exotic_single=='single':
        batch_models = model_flavors_single()
        
    return (batch_models, batch_models[process])

def model_flavors_exotic():
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    models_exotic = []
    mut_rate = 0.25

    # physics inspired
    models_exotic.append(("physi1-single-lap",     custom_svm(my_c= 100, my_gamma_end=100, myKernel="precomputed",            myDegree=1, myCoef0=+1), "precomp", "phys1",  "trad", mut_rate, "auc", "roulette", 0.0))

    # selected in prev paper
    models_exotic.append(("genHLACC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",    "rbf",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_exotic.append(("genHLAUC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='sigmoid',  myDegree=2, myCoef0=-1), "absv",    "sig",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_exotic.append(("genHLACC-pol-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=10, myKernel='poly',      myDegree=2, myCoef0=+1), "absv",    "pol",  "gene", mut_rate, "acc", "highlow", 0.5))

    # basic kernel functions
    models_exotic.append(("trad-single-rbf",     custom_svm(my_c=100, my_gamma_end=100, myKernel="rbf",                       myDegree=1, myCoef0=+1), "default", "rbf",  "trad", mut_rate, "auc", "highlow", 0.0))
    models_exotic.append(("trad-single-sig",     custom_svm(my_c=100, my_gamma_end=100, myKernel="rbf",                       myDegree=1, myCoef0=-1), "default", "sig",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-single-pol",     custom_svm(my_c=100, my_gamma_end=100, myKernel="rbf",                       myDegree=1, myCoef0=+1), "default", "pol",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-single-lin",     custom_svm(my_c=100, my_gamma_end=100, myKernel="rbf",                       myDegree=1, myCoef0=+1), "default", "lin",  "trad", mut_rate, "auc", "roulette", 0.0))

    # combined kernels
    models_exotic.append(("trad-sum-rbf-sig",   custom_svm(my_c= 100, my_gamma_end=100, myKernel="precomputed",               myDegree=1, myCoef0=+1), "precomp", "sum_rbf_sig", "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-sum-rbf-pol",   custom_svm(my_c= 100, my_gamma_end=100, myKernel="precomputed",               myDegree=1, myCoef0=+1), "precomp", "sum_rbf_pol", "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-sum-rbf-lin",   custom_svm(my_c= 100, my_gamma_end=100, myKernel="precomputed",               myDegree=1, myCoef0=+1), "precomp", "sum_rbf_lin", "trad", mut_rate, "auc", "roulette", 0.0))

    models_exotic.append(("trad-rbf-NOTdiv",    adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',      myDegree=3, myCoef0=+1), "absv",   "rbf",   "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-sig-NOTdiv",    adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='sigmoid',  myDegree=1, myCoef0=-1), "absv",   "sig",   "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-pol-NOTdiv",    adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',      myDegree=2, myCoef0=+1), "absv",   "pol",   "trad", mut_rate, "auc", "roulette", 0.0))
    models_exotic.append(("trad-lin-NOTdiv",    adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='linear',   myDegree=1, myCoef0=+1), "absv",   "lin",   "trad", mut_rate, "auc", "roulette", 0.0))

    return models_exotic


def model_flavors_single():
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    models_auc = []
    mut_rate = 0.5
    # different models
    models_auc.append(("rbf-svm", single_svm("rbf"),        "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 0
    models_auc.append(("poly-svm", single_svm("poly"),      "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 1
    models_auc.append(("sigmoid-svm", single_svm("sigmoid"),"deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 2
    models_auc.append(("linear-svm", linear_svm(),          "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 3    
    # models_auc.append(("bdt-svm",    bdt_svm(),             "prob", "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("bag-svm",    bag_svm(),             "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 4
    models_auc.append(("rand-forest",rand_forest(),         "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 5
    models_auc.append(("bdt-forest", bdt_forest(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 6
    models_auc.append(("bag-forest", bag_forest(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 7
    models_auc.append(("grad-forest",grad_forest(),         "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 8
    models_auc.append(("neural-net", neural_net(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 9
    models_auc.append(("k-neigh",    k_neighbors(),         "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 10
    models_auc.append(("gauss-nb",   gauss_nb(),            "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 11
    models_auc.append(("gauss-pc",   gauss_pc(),            "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 12
    models_auc.append(("log-reg",    log_reg(),             "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 13
    models_auc.append(("ridge-cl",   ridge_class(),         "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 14
    models_auc.append(("sgdc-cl",    sgdc_class(),          "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 15
    models_auc.append(("pass-agre",  pass_agre(),           "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 16
    models_auc.append(("linear-dis", linear_dis(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 17
    models_auc.append(("quad-dis",   quad_dis(),            "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 18
    
    return models_auc
