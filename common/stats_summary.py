"""
-----------------------------------------
 Authors: A. Ramirez-Morales
-----------------------------------------
"""
import numpy as np
import time

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

    area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores,time_scores = ([]),([]),([]),([]),([]),([]),([])
    n_class_scores, n_train_scores = ([]), ([])
        
    X, Y = data.dataset(sample_name=sample_name, data_set=sample_df,
                        sampling=True, split_sample=0.0)

    species = "class"
    balance = balance_name
    pos_label = 1
    neg_label = 0

    size_train = 5000
    size_test = 1000

    if balance =="half_half":
        size_train_sig = 5000
        size_train_bkg = 5000
    elif balance=="3quart_1quart":
        size_train_sig = int(5000*0.75)
        size_train_bkg = int(5000*0.25)
    elif balance=="1quart_3quart":
        size_train_sig = int(5000*0.25)
        size_train_bkg = int(5000*0.75)
    elif balance=="7oct_1oct":
        size_train_sig = int(5000*0.875)
        size_train_bkg = int(5000*0.125)
    elif balance=="1oct_7oct":
        size_train_sig = int(5000*0.125)
        size_train_bkg = int(5000*0.875)
    
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
    rkf = RepeatedKFold(n_splits = kfolds, n_repeats = n_reps, random_state = 1) # set random state=1 for reproducibility
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
            X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, indexes=GA_train_indexes)
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
            y_pred = model.predict(X_test)
            prec = precision_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            acc = accuracy_score(Y_test, y_pred)
            gmean = np.sqrt(prec * recall)
            # calculate roc-auc depending on the classifier
            if roc_area=="absv":
                y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
                TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
                area = auc(FPR,TPR)
                model.clean()
            elif roc_area=="prob":
                Y_pred_prob = model.predict_proba(X_test)[:,1]
                area = roc_auc_score(Y_test, Y_pred_prob)
            elif roc_area=="deci":
                Y_pred_dec = model.decision_function(X_test)
                area = roc_auc_score(Y_test, Y_pred_dec)
            else:
                area = 0

            end = time.time()
            time_scores    = np.append(time_scores, end-start)
            area_scores    = np.append(area_scores, area)
            prec_scores    = np.append(prec_scores, prec)
            f1_scores      = np.append(f1_scores,   f1)
            recall_scores  = np.append(recall_scores, recall)
            acc_scores     = np.append(acc_scores, acc)
            gmean_scores   = np.append(gmean_scores, gmean)
            n_class_scores = np.append(n_class_scores, n_base_class)
            n_train_scores = np.append(n_train_scores, len(X_train))
        else: # this needs to be re-checked carefully
            end = time.time()
            time_scores    = np.append(time_scores, end-start)
            area_scores    = np.append(area_scores, 0)
            prec_scores    = np.append(prec_scores, 0)
            f1_scores      = np.append(f1_scores,   0)
            recall_scores  = np.append(recall_scores, 0)
            acc_scores     = np.append(acc_scores, 0)
            gmean_scores   = np.append(gmean_scores, 0)
            n_class_scores = np.append(n_class_scores, 0)
            n_train_scores = np.append(n_train_scores, len(X_train))

    return area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores,time_scores,n_class_scores,n_train_scores



def stats_results(name, balance_name, n_cycles, kfolds, n_reps, boot_kfold ='', split_frac=0.6):
    # arrays to store the scores
    mean_auc,mean_prc,mean_f1,mean_rec,mean_acc,mean_gmn = ([]),([]),([]),([]),([]),([])
    std_auc,std_prc,std_f1,std_rec,std_acc,std_gmn = ([]),([]),([]),([]),([]),([])
    auc_values,prc_values,f1_values,rec_values,acc_values,gmn_values = [],[],[],[],[],[]
    names = []

    # load models and auc methods
    models_auc = mm.model_flavors_exotic()  # models_auc = mm.model_loader_batch("boot", name)
    
    for i in range(len(models_auc)):
        if boot_kfold == 'bootstrap':
            sys.exit("you are not ready for bootstrap")
        elif boot_kfold == 'kfold':

            auc, prc, f1, rec, acc, gmn, time, n_class, n_train = cross_validation(sample_name=name, balance_name=balance_name, is_precom=False, kernel_fcn=None, model=models_auc[i][1],  roc_area=models_auc[i][2],
                                                                                   selection=models_auc[i][3], GA_mut=models_auc[i][4], GA_score=models_auc[i][5],
                                                                                   GA_selec=models_auc[i][6], GA_coef=models_auc[i][7], kfolds=kfolds, n_reps=n_reps)
        auc_values.append(auc)
        prc_values.append(prc)
        f1_values.append(f1)
        rec_values.append(rec)
        acc_values.append(acc)
        gmn_values.append(gmn)
        
        mean_auc = np.append(mean_auc,  np.mean(auc))
        mean_prc = np.append(mean_prc,  np.mean(prc))
        mean_f1  = np.append(mean_f1,   np.mean(f1))
        mean_rec = np.append(mean_rec,  np.mean(rec))
        mean_acc = np.append(mean_acc,  np.mean(acc))
        mean_gmn = np.append(mean_gmn,  np.mean(gmn))
        
        std_auc = np.append(std_auc,  np.std(auc))
        std_prc = np.append(std_prc,  np.std(prc))
        std_f1  = np.append(std_f1,   np.std(f1))
        std_rec = np.append(std_rec,  np.std(rec))
        std_acc = np.append(std_acc,  np.std(acc))
        std_gmn = np.append(std_gmn,  np.std(gmn))
        
        # store model names, for later use in latex tables
        names.append(models_auc[i][0])
    
    # tukey tests
    tukey_auc  =  tukey_test(np.array(auc_values))
    tukey_prc  =  tukey_test(np.array(prc_values))
    tukey_f1   =  tukey_test(np.array(f1_values))  
    tukey_rec  =  tukey_test(np.array(rec_values)) 
    tukey_acc  =  tukey_test(np.array(acc_values))
    tukey_gmn  =  tukey_test(np.array(gmn_values))
                                 
    # latex tables
    f_tukey_noDiv = open('./tables/tukey_'+balance_name+'_'+boot_kfold+'_noDiv.tex', "w")
    dv.latex_table_tukey(names, False, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1,  tukey_f1,
                         mean_rec, std_rec,  tukey_rec, mean_acc, std_acc,  tukey_acc, mean_gmn, std_gmn,  tukey_gmn,  f_tukey_noDiv)
    f_tukey_noDiv.close()

    f_tukey_div = open('./tables/tukey_'+balance_name+'_'+boot_kfold+'_div.tex', "w")
    dv.latex_table_tukey(names, True, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1, tukey_f1,
                         mean_rec, std_rec,  tukey_rec, mean_acc, std_acc, tukey_acc, mean_gmn, std_gmn, tukey_gmn, f_tukey_div)
    f_tukey_div.close()
