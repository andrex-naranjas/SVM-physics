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
            df_sig = pd.read_csv(self.workpath+"/hep_data/data/events_data_dy_Z.csv")
            sig_class = np.full(int(len(df_sig.index)), 1)
            df_sig.insert(0, "class", sig_class, True)
            
            df_bkg_1 = pd.read_csv(self.workpath+"/hep_data/data/events_data_bkg_ZW.csv")
            bkg_class = np.full(int(len(df_bkg_1.index)), 0)
            df_bkg_1.insert(0, "class", bkg_class, True)

            df_bkg_2 = pd.read_csv(self.workpath+"/hep_data/data/events_data_bkg_WW.csv")
            bkg_class = np.full(int(len(df_bkg_2.index)), 0)
            df_bkg_2.insert(0, "class", bkg_class, True)

            df_bkg_3 = pd.read_csv(self.workpath+"/hep_data/data/events_data_bkg_ALL.csv")
            bkg_class = np.full(int(len(df_bkg_3.index)), 0)
            df_bkg_3.insert(0, "class", bkg_class, True)

            df_bkg_4 = pd.read_csv(self.workpath+"/hep_data/data/events_data_bkg_TTBAR.csv")
            bkg_class = np.full(int(len(df_bkg_4.index)), 0)
            df_bkg_4.insert(0, "class", bkg_class, True)
            frames = [df_sig, df_bkg_1, df_bkg_2, df_bkg_3, df_bkg_4]
            frames = [df_sig, df_bkg_4]
            data_set = pd.concat(frames)
            # print(len(data_set.index), len(df_bkg.index), len(df_sig.index) )
            # input()
        else:
            sys.exit("The sample name provided does not exist. Try again perrito!")
        return data_set

    
    def dataset(self, sample_name, balance_name="half_half", data_set=None, data_train=None, data_test=None,
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
        if sampling or split_sample==0.0 and indexes is None:
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


        size_test_sig = 1000
        size_test_bkg = 1000
        
        X_train = X_train.drop("z_eta_og", axis=1)
        X_train = X_train.drop("z_mass_og", axis=1)
        X_train = X_train.drop("z_pt_og", axis=1)
        #X_train = X_train.drop("reco_Z_masses", axis=1)

        X_test = X_test.drop("z_eta_og", axis=1)
        X_test = X_test.drop("z_mass_og", axis=1)
        X_test = X_test.drop("z_pt_og", axis=1)
        #X_test = X_test.drop("reco_Z_masses", axis=1)

        # balance train data
        X_train = X_train[~X_train.index.duplicated(keep='first')] # remove repeated indexes!
        
        y0_index = X_train[X_train['class'] ==  neg_label].index
        y1_index = X_train[X_train['class'] ==  pos_label].index
        print("-1: ", y0_index.shape, "   +1: ", y1_index.shape)
        random_y0 = np.random.choice(y0_index, int(size_train_bkg/2), replace = False)
        random_y1 = np.random.choice(y1_index, int(size_train_sig/2), replace = False)
        indexes = np.concatenate([random_y0, random_y1])
        X_train = X_train.loc[indexes]
        # X_train.to_csv('test_balaced.csv')

        # balance test data
        X_test = X_test[~X_test.index.duplicated(keep='first')] # remove repeated indexes!

        y0_index = X_test[X_test['class'] == neg_label].index
        y1_index = X_test[X_test['class'] == pos_label].index
        random_y0 = np.random.choice(y0_index, int(size_test_bkg/2), replace = False)
        random_y1 = np.random.choice(y1_index, int(size_test_sig/2), replace = False)
        indexes = np.concatenate([random_y0, random_y1])
        X_test = X_test.loc[indexes]

        return X_train, Y_train, X_test, Y_test


    def dataset_index(self, X, Y, split_indexes=None, train_test=False):
        '''
        Method to ...
        '''

        if not train_test:
            X_train, X_test, Y_train, Y_test = self.indexes_split(X, Y, split_indexes=split_indexes, train_test=train_test)
        else:
            X_train, X_test, Y_train, Y_test = self.indexes_split(X_train, Y_train, X_test, Y_test,
                                                                      split_indexes=split_indexes, train_test=train_test)

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
            if sig_back=="sig":   data_set = pd.read_csv(self.workpath+"/hep_data/data/events_data_dy_Z.csv")
            elif sig_back=="bkg": data_set = pd.read_csv(self.workpath+"/hep_data/data/events_data_bkg_ZW.csv")

        #print("super perrito", type(data_set.shape[0]))
        if not sampling:
            data_set = resample(data_set, replace = False, n_samples = data_set.shape[0], random_state = 3)

        data_set = data_set.copy()
        
        Y = data_set["class"]
        # data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set), # not doing this for gaussian distributions!!
                                columns = list(data_set.columns))
        X = data_set #.drop("class", axis=1)
        return X, Y
