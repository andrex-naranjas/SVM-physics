# author: Andres Ramirez-Morales
# usage: python3 scripts/simple_dy_training.py DY
import sys
import numpy as np
import pandas as pd
import datetime

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import auc,roc_auc_score
from sklearn.metrics import accuracy_score,precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# framework includes
from common.data_preparation import data_preparation
from common.svm_methods import RBFPrecomputed
from common.boosted_svm import BoostedSVM
import common.model_maker as mm
import data.data_utils as du
import common.stats_summary as ss


if len(sys.argv) != 2:
    sys.exit("Provide data sample name. Try again!")

sample_input = sys.argv[1]

# make directories
sample_list = ["half_half", "1quart_3quart", "3quart_1quart", "1dec_9dec", "9dec_1dec"]
#sample_list = ["half_half"]
du.make_directories(sample_list)

# kernel selection
kernel_list = ["linear", "poly", "rbf", "sigmoid"]
myKernel = "rbf"
sample_input = sys.argv[1]

data = data_preparation(drop_class=False)

n_cycles = 10
k_folds  = 5
n_reps   = 10

for name in sample_list:
    print("Analysing sample: ", name)
    
    # X_train, Y_train, X_test, Y_test = \
    #     data.dataset(sample_name=sample_input, balance_name=name,
    #                  sampling=False, split_sample=0.3)

    start = datetime.datetime.now()
    # kfold cross-validation
    #ss.stats_results(sample_input, balance_name=name, n_cycles=n_cycles, kfolds=k_folds, n_reps=n_reps, boot_kfold ="kfold")
    # wilcoxon stats
    ss.statistical_tests(sample_name=name)
    end = datetime.datetime.now()
    elapsed_time = end - start

    pos_label = 1
    neg_label = 0
    print("END")

    # model = BoostedSVM(C=100000, gammaEnd=10, myKernel='rbf', myDegree=3, myCoef0=+1, Diversity=False, early_stop=False, debug=True)
    # model.fit(X_train.drop("class", axis=1), X_train["class"])

    # #Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
    # # Y_pred_dec = model.decision_thresholds(X_test.drop(species, axis=1), glob_dec=True)
    # Y_pred_dec = model.predict(X_test.drop("class", axis=1))
    # Y_pred = model.predict(X_test.drop("class", axis=1))
    # Y_test_p = X_test["class"]
    # Y_test   = X_test["class"]

    # acc = accuracy_score(Y_test_p, Y_pred)
    # auc = roc_auc_score(Y_test, Y_pred_dec)
    # prc_p = precision_score(Y_test_p, Y_pred, pos_label=pos_label)
    # prc_n = precision_score(Y_test_p, Y_pred, pos_label=neg_label)
    # print("Testing accuracy linear kernel BALANCED", acc, prc_p, prc_n, auc)
