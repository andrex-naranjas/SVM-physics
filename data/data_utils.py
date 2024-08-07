'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
# utilities module
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math

# sklearn utils
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample

# framework includes
from common.data_preparation import data_preparation
from common.boosted_svm import BoostedSVM
from common.genetic_selection import GeneticSelection
from common import model_maker as mm


def make_directories(sample_list):
    # makes a directory for each dataset
    for item in sample_list:
        try:
            os.makedirs('output/{}'.format(item))
        except FileExistsError:
            pass

def cv_scores(model, x,y):
    scores = cross_val_score(model, x, y, cv=5)
    print("Cross-validation score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return ["%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)]


def cv_metrics(model, X, y):
    # Makeshift metric for predictors

    X = X.values
    y = y.values

    kf = KFold(n_splits = 5, shuffle = True)

    acc_scores    = np.array([])
    prec_scores   = np.array([])
    recall_scores = np.array([])
    f1_scores     = np.array([])

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)

        acc = accuracy_score(y_test, y_predicted)
        prec = precision_score(y_test, y_predicted)
        recall = recall_score(y_test, y_predicted)
        f1 = f1_score(y_test, y_predicted)

        acc_scores = np.append(acc_scores, acc)
        prec_scores = np.append(prec_scores, prec)
        recall_scores = np.append(recall_scores, recall)
        f1_scores = np.append(f1_scores, f1)

    print("Cross-validation Accuracy Score: %0.2f (+/- %0.2f)" % (acc_scores.mean(), acc_scores.std() * 2))
    print("Cross-validation Precision Score: %0.2f (+/- %0.2f)" % (prec_scores.mean(), prec_scores.std() * 2))
    print("Cross-validation Recall Score: %0.2f (+/- %0.2f)" % (recall_scores.mean(), recall_scores.std() * 2))
    print("Cross-validation F1 Score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std() * 2))

def generate_report(y_val, y_pred, verbose):
    acc    = round(accuracy_score(y_val, y_pred) * 100, 2)
    prec   = round(precision_score(y_val, y_pred) * 100 ,2)
    recall = round(recall_score(y_val, y_pred) * 100, 2)
    f1     = round(f1_score(y_val, y_pred) * 100, 2)

    if verbose:
        print('Accuracy = ', acc)
        print('Precision = ', prec)
        print('Recall = ', recall)
        print('f1_score =', f1)

    return [acc, prec, recall, f1]

def generate_auc_roc_curve(sample, model, X_val, Y_test, name):
    Y_pred_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    auc = round(roc_auc_score(Y_test, Y_pred_prob) *100 ,2)
    string_model= str(model)
    #plt.plot(fpr, tpr, label = 'AUC ROC ' + string_model[:3] + '=' + str(auc))
    #plt.legend(loc = 4)
    #plt.savefig(name+'.pdf')
    output = pd.DataFrame({'False positive rate': fpr,'True positive rate': tpr, 'Area': auc})
    output.to_csv('output/' + sample +  '/' + string_model[:3] + 'roc.csv', index=False)
    return

def metrics(sample, name, method, X_train, Y_train, Y_test, X_test, Y_pred):
    generate_auc_roc_curve(sample, method, X_test,Y_test, name)
    print('\n '+name+': ')
    return cv_scores(method, X_train, Y_train) + generate_report(Y_test, Y_pred, verbose=True)


def error_number(sample_name, myC, myGammaIni, train_test):
    # function to get average errors via bootstrap, for 1-n classifiers
    
    print('Start of error number')
    # fetch data_frame without preparation
    data_df   = data_preparation()
    if not train_test: sample_df = data_df.fetch_data(sample_name)
    else: sample_train_df, sample_test_df = data_df.fetch_data(sample_name)

    n_samples = 795
    selection = 'trad'
    selection = 'gene'
    GA_mut = 0.25
    GA_score = "auc" # "acc"
    GA_selec = "highlow"
    GA_coef = 0.5
    roc_area="absv"

    # run AdaBoostSVM (train the model)
    model = mm.adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1)
    model = mm.adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1)   #"genHLACC-rbf-NOTdiv",   "acc"
    model = mm.adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1)  # "genHLAUC-sig-YESdiv", "auc"
    model = mm.adaboost_svm(div_flag=True, my_c=10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1)   # "genHLACC-pol-YESdiv", "acc"        
        
    # prepare bootstrap sample
    total = []
    number = ([])

    for i in range(10): # arbitrary number of samples to produce

        data = data_preparation()
        
        if not train_test:
            # sampled_data = resample(sample_df, replace = True, n_samples = 500, random_state = None)
            # X_train, Y_train, X_test, Y_test = data.dataset(sample_name, data_set=sampled_data, sampling=True, split_sample=0.4) #, train_test=True

            #sampled_data_train = resample(sample_df, replace = True, n_samples = n_samples, random_state = None)
            sampled_data_train = resample(sample_df, replace = False, n_samples = n_samples, random_state = 1)
            print('random state: ', i)
            
            if selection == 'trad':
                # test data are the complement of full input data that is not considered for training
                sampled_train_no_dup = sampled_data_train.drop_duplicates(keep=False)
                sampled_data_test    = pd.concat([sample_df, sampled_train_no_dup]).drop_duplicates(keep=False)
                
                X_train, Y_train = data.dataset(sample_name=sample_name, data_set=sampled_data_train,
                                                sampling=True, split_sample=0.0)
                X_test, Y_test   = data.dataset(sample_name=sample_name, data_set=sampled_data_test,
                                                sampling=True, split_sample=0.0)
                
                sample_df = data_df.fetch_data(sample_name)
                X_train, Y_train, X_test, Y_test = data.dataset(sample_name, data_set=sample_df, sampling=False, split_sample=0.4) #, train_test=True
                print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
                
            elif selection == 'gene': # genetic selection
                train_indexes = sampled_data_train.index
                X,Y = data.dataset(sample_name=sample_name, data_set = sample_df, sampling = True)
                X_train, X_test, Y_train, Y_test = data.indexes_split(X, Y, split_indexes=train_indexes, train_test=train_test)
                print(len(X_train.index),  len(Y_train.index), 'X_train, Y_train sizes')
                # sample_df = data_df.fetch_data(sample_name)
                # X_train, Y_train, X_test, Y_test = data.dataset(sample_name, data_set=sample_df, sampling=False, split_sample=0.4) #, train_test=True
                # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

                GA_selection = genetic_selection(model, roc_area, X_train, Y_train, X_test, Y_test,
                                                 pop_size=10, chrom_len=int(len(Y_train.index)*0.20), n_gen=50,
                                                 coef=GA_coef, mut_rate=GA_mut, score_type=GA_score, selec_type=GA_selec)
                GA_selection.execute()
                GA_train_indexes = GA_selection.best_population()
                X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, indexes=GA_train_indexes)
            
        else:
            sampled_data_train = resample(sample_train_df, replace = True, n_samples = 5000,  random_state = None)
            sampled_data_test  = resample(sample_test_df,  replace = True, n_samples = 10000, random_state = None)
            X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, data_set='',
                                                            data_train=sampled_data_train, data_test = sampled_data_test,
                                                            sampling=True, split_sample=0.4) #, train_test=True)

            
        model.fit(X_train, Y_train)
        # compute test samples
        test_number = model.number_class(X_test)
        #model.clean()
        number = np.append(number, [len(test_number)])
        error = ([])
        for i in range(len(test_number)):
            error_d = 0
            error_d = (test_number[i] != Y_test).mean()
            error   = np.append(error, [round(error_d * 100, 2)])

            total.append(error)

    total = np.array(total)

    # complete totalarray with nan's for dimension consistency
    total_final = []
    for i in range(len(total)):
        if(len(total[i]) < np.amax(number)):
            for _ in range(int(np.amax(number) - len(total[i]))):
                total[i] = np.append(total[i], [np.nan])

        total_final.append(total[i])

    total_final = np.array(total_final)
    final_final = np.nanmean(total_final,axis=0)

    return pd.DataFrame(final_final,np.arange(np.amax(number)))


def grid_param_gauss(train_x, train_y, test_x, test_y, sigmin=-5, sigmax=5, cmin=0, cmax=6, my_kernel='rbf', train_test='train'):
    # grid svm-hyperparameters (sigma and C) to explore test errors

    # inverted limits, to acommodate the manner at which the arrays are stored and plotted as a matrix
    # sigmin = -5    sigmax = 5    cmin = 0    cmax = 6
    # log_step_c     = np.logspace(cmax,    cmin,20,endpoint=True,base=math.e)
    # log_step_sigma = np.logspace(sigmax,sigmin,20,endpoint=True,base=math.e)
    my_coef = 1
    if my_kernel == 'rbf':
        sigmax,sigmin=100.0,0.0
        cmax,cmin=100.0,0.0
    elif my_kernel == 'sigmoid':
        sigmax,sigmin=0.1,0.0
        cmax,cmin=100.0,0.0
        my_coef = -1        
    elif my_kernel == 'poly' or my_kernel == 'linear':
        sigmax,sigmin=0.1,0.0
        cmax,cmin=10.,0.0
        
    log_step_c     = np.linspace(cmax,    cmin,10,endpoint=False)
    log_step_sigma = np.linspace(sigmax,sigmin,10,endpoint=False)

    error_matrix = []
    for i in range(len(log_step_c)): # C loop
        print('************************************************************************')
        errors = ([])
        for j in range(len(log_step_sigma)): # sigma loop
            #my_gamma=1/(2*((log_step_sigma[j])**2))
            my_gamma = log_step_sigma[j]
            my_c = log_step_c[i]
            #my_c = 10
            svc = SVC(C=my_c, kernel=my_kernel, degree=2, gamma=my_gamma, coef0=my_coef, shrinking = True, probability = True, tol = 0.001)
            svc.fit(train_x, train_y)

            if train_test == 'train':
                pred_y = svc.predict(train_x)
                acc, prec, recall, f1 = generate_report(train_y, pred_y, verbose=False)
            elif train_test == 'test':
                pred_y = svc.predict(test_x)
                acc, prec, recall, f1 = generate_report(test_y, pred_y, verbose=False)

            print('creating matrix element:', i,j, round(log_step_c[i],2), round(log_step_sigma[j],2), round(my_gamma,2),
                  round((100-acc),2), my_kernel)

            errors = np.append(errors,[(0.01)*(100-acc)])

        error_matrix.append(errors)

    return np.array(error_matrix)


def roc_curve_adaboost(Y_thresholds, Y_test):
    # function to create the TPR and FPR, for ROC curve
    # check data format
    if type(Y_test) != type(np.array([])):
        Y_test = Y_test.values

    TPR_list, FPR_list = [], []
    for i in range(Y_thresholds.shape[0]):
        tp,fn,tn,fp=0,0,0,0
        for j in range(Y_thresholds.shape[1]):             
            if(Y_test[j] == 1  and Y_thresholds[i][j] ==  1):  tp+=1
            if(Y_test[j] == 1  and Y_thresholds[i][j] == -1):  fn+=1
            if(Y_test[j] == -1 and Y_thresholds[i][j] == -1):  tn+=1
            if(Y_test[j] == -1 and Y_thresholds[i][j] ==  1):  fp+=1

        TPR_list.append( tp/(tp+fn) )
        FPR_list.append( fp/(tn+fp) )

    # sort the first list and map ordered indexes to the second list
    FPR_list, TPR_list = zip(*sorted(zip(FPR_list, TPR_list)))
    TPR = np.array(TPR_list)
    FPR = np.array(FPR_list)

    return TPR,FPR
