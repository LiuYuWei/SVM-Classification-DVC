import os, sys

import pandas as pd
import numpy as np

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class DataPreprocessing():
    
    def __init__(self):
        self.df = pd.DataFrame()
        self.x_train = ''
        self.x_test = ''
        self.y_train = ''
        self.y_test = ''
        self.sc = ''

    def extract(self, data_path):
        self.df = pd.read_csv(data_path, encoding='utf-8')
        print("extract data from {} , data shape {}".format(data_path, self.df.shape))
        return self.df
        
    def transform(self):
        data, label = self.SplitDataLabel()
        self.x_train, self.x_test, self.y_train, self.y_test = \
                                    self.train_test_split(data, label, train_size=0.8, test_size=0.5)
        self.sc, self.x_train, self.x_test = self.standard()
        return self.x_train, self.x_test, self.y_train, self.y_test, self.sc
    
    def loadTo(self, output_path):
        print('saving data to data/prepared/')
        dataPreprocess.file_to_pkl(self.x_train, output_path+'dataPreprocess_x_train.pkl')
        dataPreprocess.file_to_pkl(self.y_train, output_path+'dataPreprocess_y_train.pkl')
        dataPreprocess.file_to_pkl(self.x_test, output_path+'dataPreprocess_x_test.pkl')
        dataPreprocess.file_to_pkl(self.y_test, output_path+'dataPreprocess_y_test.pkl')
        dataPreprocess.file_to_pkl(self.sc, output_path+'dataPreprocess_sc.pkl')

    def SplitDataLabel(self):
        #Spliting the dataset in independent and dependent variables
        data = self.df.iloc[:,0:4].values
        label = self.df.iloc[:,4].values
        return data, label

    def train_test_split(self, data, label, train_size=0.8, test_size=0.5):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, label, test_size = train_size, random_state = 42)
        return self.x_train, self.x_test, self.y_train, self.y_test 
    
    # Feature Scaling to bring the variable in a single scale
    def standard(self):
        self.sc = StandardScaler()
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)
        return self.sc, self.x_train, self.x_test

    @staticmethod
    def file_to_pkl(be_pickled_file, path):
        to_pickle = open(path, 'wb')
        pickle.dump(be_pickled_file, to_pickle, protocol=4)
        to_pickle.close()

if __name__ == '__main__':
    mkdir_p(os.path.join('data', 'prepared'))
    dataPreprocess = DataPreprocessing()
    dataPreprocess.extract(data_path = sys.argv[1])
    dataPreprocess.transform()
    dataPreprocess.loadTo(output_path = sys.argv[2])
