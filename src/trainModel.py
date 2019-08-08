import sys
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

class svmModel:

    def __init__(self):
        self.x_train = ''
        self.y_train = ''
        self.sc = ''
        self.model = ''
    
    def loadData(self,input_path):
        with open(os.path.join(input_path, 'dataPreprocess_x_train.pkl'), 'rb') as fd:
            self.x_train = pickle.load(fd)
        with open(os.path.join(input_path, 'dataPreprocess_y_train.pkl'), 'rb') as fd:
            self.y_train = pickle.load(fd)
        print("Input x_train and y_train.")
        return self.x_train, self.y_train
    
    def trainModel(self):
        self.model = SVC(kernel = 'linear', random_state = 0)
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def loadTo(self,output_path):
        svmModel.file_to_pkl(self.model, output_path)
        print("Output the SVM model.")

    @staticmethod
    def file_to_pkl(be_pickled_file, path):
        to_pickle = open(path, 'wb')
        pickle.dump(be_pickled_file, to_pickle, protocol=4)
        to_pickle.close()


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]

    svm = svmModel()
    svm.x_train, svm.y_train = svm.loadData(input_path = input)
    svm.model = svm.trainModel()
    svm.loadTo(output_path = output)