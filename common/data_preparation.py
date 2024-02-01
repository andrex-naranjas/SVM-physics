"""
-----------------------------------------
 Authors: A. Ramirez-Morales
-----------------------------------------
"""
# data preparation module
import sys
import pandas as pd
import numpy as np
import uproot # ROOT format data
# sklearn utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
# framework includes
#import data_visualization as dv

class data_preparation:
    '''
    Class to prepare data for training and testing a ML model
    '''
    def __init__(self, path=".", drop_class=True, GA_selection=False):
        self.m_drop_class = drop_class
        self.workpath = path
        self.genetic = GA_selection

    def fetch_data(self, sample):
        '''
        Method to fetch data
        Returns: a pandas dataframe
        '''        
        if sample == "DY":
            df_sig = pd.read_csv(self.workpath+"/hep_data/events_data_dy.csv")
            sig_class = np.full(int(len(df_sig.index)), 1)
            df_sig.insert(0, "class", sig_class, True)
            df_bkg = pd.read_csv(self.workpath+"/hep_data/events_data_bkg_ZW.csv")
            bkg_class = np.full(int(len(df_bkg.index)), -1)
            df_bkg.insert(0, "class", bkg_class, True)
            frames = [df_sig, df_bkg]
            data_set = pd.concat(frames)
            # print(len(data_set.index), len(df_bkg.index), len(df_sig.index) )
            # input()
        else:
            sys.exit("The sample name provided does not exist. Try again!")
        return data_set

    
    def dataset(self, sample_name, data_set=None, data_train=None, data_test=None,
                sampling=False, split_sample=0, indexes = None):
        '''
        Method to call data
        '''
        train_test = False # to check if data is divided
        # if sampling=True, sampling is done outside,sample is fetched externally

        # fetch data_set if NOT externally provided
        if not sampling:
            data_temp = self.fetch_data(sample_name)
            train_test = type(data_temp) is tuple
            if not train_test:
                data_set = data_temp
            else: # there is separate data samples for training and testing
                data_train, data_test = data_temp

        # prepare data
        if sample_name == "DY":
            X,Y = self.dy_data(data_set, sampling, sample_name=sample_name)
        else:
            sys.exit("The sample name provided does not exist. Try again!")
            
        if not sampling:
            if train_test:
                print(X_train.head())#, Y.head())
                print(Y_train.head())#, Y.head())
            else:
                print(X.head())#, Y.head())
                print(Y.head())#, Y.head())

         # return X,Y without any spliting (for bootstrap and kfold-CV)
        if sampling or split_sample==0.0:
            if not train_test:
                return X,Y
            else:
                return X_train, Y_train, X_test, Y_test
            
        # divide sample into train and test sample
        if indexes is None:
            if not train_test:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_sample, random_state=2) # split_sample
        else:
            if not train_test:
                X_train, X_test, Y_train, Y_test = self.indexes_split(X, Y, split_indexes=indexes, train_test=train_test)
            else:
                X_train, X_test, Y_train, Y_test = self.indexes_split(X_train, Y_train, X_test, Y_test,
                                                                      split_indexes=indexes, train_test=train_test)
                
        return X_train, Y_train, X_test, Y_test

    
    def indexes_split(self, X, Y, x_test=None, y_test=None, split_indexes=None, train_test=False):
        '''
        Method to split train and test data given train indexes
        '''
        if not train_test:
            total_indexes = np.array(X.index).tolist()        
            train_indexes = split_indexes.tolist()
            test_indexes  = list(set(total_indexes) - set(train_indexes))
        else:
            train_indexes = split_indexes.tolist()
            test_indexes  = split_indexes.tolist()

        X_train = X.loc[train_indexes]
        Y_train = Y.loc[train_indexes]
        X_test  = X.loc[test_indexes]
        Y_test  = Y.loc[test_indexes]
        return X_train, X_test, Y_train, Y_test

    
    def dy_data(self, data_set=None, sampling=False, sample_name="toy", sig_back='sig'):
        '''
        Method to prepare data
        '''
        if data_set is None:
            if sig_back=="sig":   data_set = pd.read_csv(self.workpath+"/hep_data/events_data_dy.csv")
            elif sig_back=="bkg": data_set = pd.read_csv(self.workpath+"/hep_data/events_data_bkg_ZW.csv")
            
        if not sampling:
            data_set = resample(data_set, replace = False, n_samples = 2000, random_state = 3)

        data_set = data_set.copy()
        
        Y = data_set["class"]
        # data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set), # not doing this for gaussian distributions!!
                                columns = list(data_set.columns))
        X = data_set #.drop("class", axis=1)
        return X, Y
