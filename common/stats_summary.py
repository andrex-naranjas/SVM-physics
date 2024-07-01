"""
-----------------------------------------
 Authors: A. Ramirez-Morales
-----------------------------------------
"""
import os
import sys
import numpy as np
import time
import pandas as pd

from sklearn.metrics import accuracy_score,auc,precision_score,roc_auc_score,f1_score,recall_score
from sklearn.model_selection import RepeatedKFold

# framework includes
from common.data_preparation import data_preparation
from common.common_methods import roc_curve_adaboost
from common.svm_methods import compute_kernel
import common.model_maker as mm
import data.data_visualization as dv
from common.genetic_selection import GeneticSelection

# statsmodel includes
from statsmodels.stats.multicomp import pairwise_tukeyhsd,MultiComparison
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.contingency_tables import mcnemar

# scypi includes
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import f_oneway
from scipy.stats import norm,normaltest,shapiro,chisquare,kstest



def tukey_test(score_array):

    # create INTEGER indexes to label scores
    index_array = []
    for i in range(len(score_array)):
        index_dummy = np.array([(int(i+1)) for j in range(len(score_array[i]))])
        index_array.append(index_dummy)

    # transform arrays to tuples
    score_tuple = tuple(map(tuple, score_array))
    index_tuple = tuple(map(tuple, np.array(index_array)))

    # format data for tukey function
    indexes= np.concatenate(index_tuple, axis=0)                            
    values = np.concatenate(score_tuple, axis=0)                            
    data   = {'means':values, 'group':indexes}

    # perform the pairwise tukey test
    MultiComp2 = MultiComparison(data['means'], data['group'])
    print(MultiComp2.tukeyhsd(0.05).summary())
    return MultiComp2.tukeyhsd(0.05)


def cross_validation(sample_name, balance_name, model, is_precom, kernel_fcn, roc_area, selection, GA_mut=0.25, GA_score='', GA_selec='', GA_coef=0.5, kfolds=1, n_reps=1, path='.'):
    
    # fetch data_frame without preparation
    data = data_preparation(path)
    sample_df_temp = data.fetch_data(sample_name)
    train_test = type(sample_df_temp) is tuple  # are the data already splitted?
    if not train_test:
        sample_df = sample_df_temp
    else:
        sample_train_df, sample_test_df = sample_df_temp

    # area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores,time_scores = ([]),([]),([]),([]),([]),([]),([])
    area_scores     = ([])
    prec_pos_scores = ([])
    prec_neg_scores = ([])
    acc_scores      = ([])
    n_class_scores  = ([])
    n_train_scores  = ([])
    time_scores     = ([])
        
    X, Y = data.dataset(sample_name=sample_name, data_set=sample_df,
                        sampling=True, split_sample=0.0)
    species = "class"
    balance = balance_name
    pos_label = 1
    neg_label = 0

    size_train = 5000
    size_test = 1000

    if balance =="half_half":
        size_train_sig = int(size_train*0.5)
        size_train_bkg = int(size_train*0.5)
    elif balance=="3quart_1quart":
        size_train_sig = int(size_train*0.75)
        size_train_bkg = int(size_train*0.25)
    elif balance=="1quart_3quart":
        size_train_sig = int(size_train*0.25)
        size_train_bkg = int(size_train*0.75)
    elif balance=="9dec_1dec":
        size_train_sig = int(size_train*0.9)
        size_train_bkg = int(size_train*0.1)
    elif balance=="1dec_9dec":
        size_train_sig = int(size_train*0.1)
        size_train_bkg = int(size_train*0.9)
    
    X = X.drop("z_eta_og", axis=1)
    X = X.drop("z_mass_og", axis=1)
    X = X.drop("z_pt_og", axis=1)
    #X_train = X_train.drop("reco_Z_masses", axis=1)
    
    # balance train data
    X = X[~X.index.duplicated(keep='first')] # remove repeated indexes!
    y0_index = X[X['class'] ==  neg_label].index
    y1_index = X[X['class'] ==  pos_label].index
    print("-1: ", y0_index.shape, "   +1: ", y1_index.shape)
    random_y0 = np.random.choice(y0_index, int(size_train_bkg/2), replace = False)
    random_y1 = np.random.choice(y1_index, int(size_train_sig/2), replace = False)
    indexes = np.concatenate([random_y0, random_y1])
    X = X.loc[indexes].reset_index(drop=True)
    Y = X["class"]
    X = X.drop("class", axis=1)

    # n-k fold cross validation, n_cycles = n_splits * n_repeats
    rkf = RepeatedKFold(n_splits = kfolds, n_repeats = n_reps, random_state = None) # set random state=1 for reproducibility
    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
        start = time.time()
        # keep the chromosome size within the range [100,1000]
        sample_chromn_len = int(len(Y_train)*0.25)
        if sample_chromn_len > 1000:
            sample_chromn_len = 1000
        elif sample_chromn_len < 100:
            sample_chromn_len = 100
            
        if selection == 'gene': # genetic selection
            GA_selection = GeneticSelection(model, roc_area, is_precom, kernel_fcn, X_train, Y_train, X_test, Y_test,
                                            pop_size=10, chrom_len=sample_chromn_len, n_gen=50, coef=GA_coef,
                                            mut_rate=GA_mut, score_type=GA_score, selec_type=GA_selec)
            GA_selection.execute()
            GA_train_indexes = GA_selection.best_population()            
            X_train, Y_train, X_test, Y_test = data.dataset_index(X, Y, split_indexes=GA_train_indexes)
            print(len(X_train), len(Y_test), len(GA_train_indexes), 'important check for GA outcome')
            print(len(Y_train[Y_train==1]), 'important check for GA outcome')

        if is_precom=="precomp": # pre-compute the kernel matrices if requested
            matrix_train = compute_kernel(kernel_fcn, X_train)
            X_test = compute_kernel(kernel_fcn, X_train, X_test)
            model.fit(matrix_train, Y_train)
        else:
            model.fit(X_train, Y_train)

        n_base_class = 0
        no_zero_classifiers = True
        if roc_area=="absv":
            n_base_class = model.n_classifiers
            if n_base_class==0:
                no_zero_classifiers = False
                
        if no_zero_classifiers:
            y_pred   = model.predict(X_test)
            prec_pos = precision_score(Y_test, y_pred, pos_label=pos_label)
            prec_neg = precision_score(Y_test, y_pred, pos_label=neg_label)
            acc      = accuracy_score(Y_test, y_pred)
           
            # calculate roc-auc depending on the classifier
            if roc_area=="absv":
                # y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
                # TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
                # area = auc(FPR,TPR)
                area = roc_auc_score(Y_test, y_pred)
                model.clean()
            elif roc_area=="prob":
                # Y_pred_prob = model.predict_proba(X_test)[:,1]
                # area = roc_auc_score(Y_test, Y_pred_prob)
                area = roc_auc_score(Y_test, y_pred)
                model.clean()
            elif roc_area=="deci":
                # Y_pred_dec = model.decision_function(X_test)
                # area = roc_auc_score(Y_test, Y_pred_dec)
                area = roc_auc_score(Y_test, y_pred)
                model.clean()
            else:
                area = 0

            end = time.time()
            time_scores     = np.append(time_scores, end-start)
            
            area_scores     = np.append(area_scores, area)
            prec_pos_scores = np.append(prec_pos_scores, prec_pos)
            prec_neg_scores = np.append(prec_neg_scores, prec_neg)
            acc_scores      = np.append(acc_scores, acc)

            n_class_scores = np.append(n_class_scores, n_base_class)
            n_train_scores = np.append(n_train_scores, len(X_train))
        else: # this needs to be re-checked carefully
            end = time.time()
            time_scores     = np.append(time_scores, end-start)

            area_scores     = np.append(area_scores, 0)
            prec_pos_scores = np.append(prec_pos_scores, prec_pos)
            prec_neg_scores = np.append(prec_neg_scores, prec_neg)
            acc_scores      = np.append(acc_scores, acc)
            
            n_class_scores = np.append(n_class_scores, 0)
            n_train_scores = np.append(n_train_scores, len(X_train))

    del model
    return area_scores,prec_pos_scores,prec_neg_scores,acc_scores,time_scores,n_class_scores,n_train_scores



def stats_results(name, balance_name, n_cycles, kfolds, n_reps, boot_kfold ='', split_frac=0.6):
    
    # arrays to store the scores
    auc_values     , mean_auc     , std_auc      = ([]),([]),([])
    prec_pos_values, mean_prec_pos, std_prec_pos = ([]),([]),([])
    prec_neg_values, mean_prec_neg, std_prec_neg = ([]),([]),([])
    acc_values     , mean_acc     , std_acc      = ([]),([]),([])
    n_class_values , mean_n_class , std_n_class  = ([]),([]),([])
    n_train_values , mean_n_train , std_n_train  = ([]),([]),([])
    time_values    , mean_time    , std_time     = ([]),([]),([])
    names = []

    path = "."
    
    # load models and auc methods
    models_auc = mm.model_flavors_exotic()  # models_auc = mm.model_loader_batch("boot", name)
    
    for i in range(len(models_auc)):
        if boot_kfold == 'bootstrap':
            sys.exit("you are not ready for bootstrap")
        elif boot_kfold == 'kfold':

            auc, prec_pos, prec_neg, acc, time, n_class, n_train = cross_validation(sample_name=name,
                                                                                    balance_name=balance_name,
                                                                                    model=models_auc[i][1],
                                                                                    is_precom=models_auc[i][2],
                                                                                    kernel_fcn=models_auc[i][3],
                                                                                    roc_area="deci",
                                                                                    selection=models_auc[i][3+1],
                                                                                    GA_mut=models_auc[i][4+1],
                                                                                    GA_score=models_auc[i][5+1],
                                                                                    GA_selec=models_auc[i][6+1],
                                                                                    GA_coef=models_auc[i][7+1],
                                                                                    kfolds=kfolds, n_reps=n_reps)
            col_auc      = pd.DataFrame(data=auc,       columns=["auc"])
            col_prec_pos = pd.DataFrame(data=prec_pos,  columns=["prec_pos"])
            col_prec_neg = pd.DataFrame(data=prec_neg,  columns=["prec_neg"])
            col_acc      = pd.DataFrame(data=acc,       columns=["acc"])
            col_time     = pd.DataFrame(data=time,      columns=["time"])
            col_base     = pd.DataFrame(data=n_class,   columns=["n_base"])
            col_size     = pd.DataFrame(data=n_train,   columns=["n_train"])

            df = pd.concat([col_auc["auc"], col_prec_pos["prec_pos"], col_prec_neg["prec_neg"], col_acc["acc"], col_time["time"], col_base["n_base"], col_size["n_train"]],
                           axis=1, keys=["auc", "prec_pos", "prec_neg", "acc", "time", "n_base", "n_train"])

            dir_name_csv = path + "/results/stats_results/"+name+"/"+boot_kfold+"/"

            if not os.path.exists(dir_name_csv):
                os.makedirs(dir_name_csv)
    
            name_csv = dir_name_csv + models_auc[i][0]+"_"+boot_kfold+".csv" 
            df.to_csv(str(name_csv), index=False)

            
        auc_values.append(auc)
        prec_pos_values.append(prec_pos)
        prec_neg_values.append(prec_neg)
        acc_values.append(acc)
        
        mean_auc = np.append(mean_auc,  np.mean(auc))
        mean_prec_pos = np.append(mean_prec_pos,  np.mean(prec_pos))
        mean_prec_neg = np.append(mean_prec_neg,  np.mean(prec_neg))
        mean_acc = np.append(mean_acc,  np.mean(acc))
        
        std_auc = np.append(std_auc,  np.std(auc))
        std_prec_pos = np.append(std_prec_pos,  np.std(prec_pos))
        std_prec_pos = np.append(std_prec_neg,  np.std(prec_neg))
        std_acc = np.append(std_acc,  np.std(acc))
        
        # store model names, for later use in latex tables
        names.append(models_auc[i][0])
    
    print(np.array(prec_pos_values))
    print(np.mean(np.array(prec_pos_values)), "mean value")


def statistical_tests(sample_name='half_half', class_interest=['physi1-single-lap'], metric='AUC', stats_type='student', boot_kfold='kfold'):

    ensembles_pvalues = []
    selected_values   = []
    selected_errors   = []

    # make the list of classifier flavors,set here the top classifier we want to compare against to and show in tables
    for i_name in range(len(class_interest)):
        # arrays to store the scores
        auc_values     , mean_auc     , std_auc,      student_auc      = ([]),([]),([]),([])
        prec_pos_values, mean_prec_pos, std_prec_pos, student_prec_pos = ([]),([]),([]),([])
        prec_neg_values, mean_prec_neg, std_prec_neg, student_prec_neg = ([]),([]),([]),([])
        acc_values     , mean_acc     , std_acc,      student_acc      = ([]),([]),([]),([])

        mean_metric = ([])
        std_metric = ([])

        flavor_names = []
        flavor_names.append(class_interest[i_name])
        
        for i in range(len(mm.model_loader_batch(0, exotic_single='exotic')[0])):
            flavor_names.append(mm.model_loader_batch(0, exotic_single='exotic')[0][i][0])

        directory = './results/stats_results/'+sample_name+'/'+ boot_kfold
        nClass  = len(flavor_names)
        f_names = []
        matrix  = []
            
        for i in range(len(flavor_names)):
            input_data = pd.read_csv(directory+'/'+flavor_names[i]+'_'+boot_kfold+'.csv')

            print(flavor_names[i], i)
            auc      = np.array(input_data['auc'])
            prec_pos = np.array(input_data['prec_pos'])
            prec_neg = np.array(input_data['prec_neg'])
            acc      = np.array(input_data['acc'])

            auc_values.append(auc)
            prec_pos_values.append(prec_pos)
            prec_neg_values.append(prec_neg)
            acc_values.append(acc)
                
            mean_auc      = np.append(mean_auc,  np.mean(auc))
            mean_prec_pos = np.append(mean_prec_pos,  np.mean(prec_pos))
            mean_prec_neg = np.append(mean_prec_neg,  np.mean(prec_neg))
            mean_acc      = np.append(mean_acc,  np.mean(acc))
            
            std_auc      = np.append(std_auc,  np.std(auc))
            std_prec_pos = np.append(std_prec_pos,  np.std(prec_pos))
            std_prec_neg = np.append(std_prec_neg,  np.std(prec_neg))
            std_acc      = np.append(std_acc,  np.std(acc))
                
            # check normality
            # p,alpha = normal_test(auc,alpha=0.05,verbose=True)
            # dv.simple_plot(auc, pval=p, alpha_in=alpha)

        if stats_type=='student':
            for i in range(nClass):
                column = ([])
                for j in range(nClass):
                    # print(ttest_ind(auc_values[i], auc_values[j]).pvalue)
                    if(i==0 and j!=0):
                        #wilcoxon, ttest_ind
                        student_auc      = np.append(student_auc,      wilcoxon(auc_values[i], auc_values[j]).pvalue)
                        student_prec_pos = np.append(student_prec_pos, wilcoxon(prec_pos_values[i], prec_pos_values[j]).pvalue)
                        student_prec_neg = np.append(student_prec_neg, wilcoxon(prec_neg_values[i], prec_neg_values[j]).pvalue)
                        student_acc      = np.append(student_acc,      wilcoxon(acc_values[i], acc_values[j]).pvalue)
                    elif(i==0 and j==0):
                        student_auc      = np.append(student_auc, 1.)
                        student_prec_pos = np.append(student_prec_pos, 1.)
                        student_prec_neg = np.append(student_prec_neg, 1.)
                        student_acc      = np.append(student_acc, 1.)
                    if(i!=j):                        
                        pvalue = wilcoxon(auc_values[i], auc_values[j]).pvalue
                    else:
                        pvalue = 1.
                    
                    if pvalue < 0.05:
                        column = np.append(column, 1)
                    else:
                        column = np.append(column, -1)
                                                 
                matrix.append(column)
                       
            matrix = np.array(matrix)
                        
            # latex tables
            table_sample_name=sample_name+'-'+boot_kfold+'-'+stats_type
            f_student_table = open('./tables/student_'+table_sample_name+'_'+class_interest[i_name]+'.tex', "w")
            dv.latex_table_student(flavor_names, sample_name, mean_auc, std_auc, student_auc, mean_prec_pos, std_prec_pos,  student_prec_pos,
                                   mean_prec_neg, std_prec_neg, student_prec_neg,
                                   mean_acc, std_acc, student_acc, f_student_table)
            f_student_table.close()
        
        elif stats_type == 'tukey':
            # not working atm 29/06/2024
            # tukey tests
            tukey_auc  =  tukey_test(np.array(auc_values))
            counter  = 0
            counter2 = 0
            flag = True        
            column = ([])
            for k in range(len(tukey_auc.reject)):
                counter2+=1
                if counter2 == 1:
                    column = np.append(column, -1.)
                    # column = np.append(column, tukey_auc.pvalues[k])
                    if tukey_auc.reject[k]: value = 1
                    else: value = -1
                    column = np.append(column, value)
                    if nClass - counter - 2 < counter2:
                        column = np.flip(column)
                        zeros = np.zeros(nClass-len(column))
                        column = np.append(column, zeros)
                        column = np.flip(column)
                        matrix.append(column)
                        column = ([])
                        counter += 1
                        counter2 = 0
            
            last_column = np.array([-1.])
            zeros = np.zeros(nClass-len(last_column))
            last_column = np.append(zeros, last_column)
            matrix.append(last_column)
            matrix = np.array(matrix)
            #matrix = matrix.transpose()
            tukey_auc  =  tukey_test(np.array(auc_values))
            tukey_prc  =  tukey_test(np.array(prc_values))
            tukey_f1   =  tukey_test(np.array(f1_values))
            tukey_rec  =  tukey_test(np.array(rec_values)) 
            tukey_acc  =  tukey_test(np.array(acc_values))
            tukey_gmn  =  tukey_test(np.array(gmn_values))                         
            # latex tables
            sample_name+='-'+boot_kfold+'-'+stats_type
            f_tukey_table = open('./tables/tukey_'+sample_name+'_'+class_interest+'.tex', "w")
            dv.latex_table_tukey(f_names, sample_name, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1,  tukey_f1,
                                 mean_rec, std_rec, tukey_rec, mean_acc, std_acc,  tukey_acc, mean_gmn, std_gmn,  tukey_gmn,  f_tukey_table)
            f_tukey_table.close()
                            
        sigmin = 0
        sigmax = len(flavor_names)
        cmin = 0
        cmax = len(flavor_names)
        dv.plot_stats_2d(matrix, sample_name)
    
