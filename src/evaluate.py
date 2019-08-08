import sys
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix

class evaluation:

    def __init__(self):
        self.x_test = ''
        self.y_test = ''
        self.y_pred = ''
        self.sc = ''
        self.model = ''
        self.auc = ''
    
    def loadData(self, input_path, model_path):
        with open(os.path.join(input_path, 'dataPreprocess_x_test.pkl'), 'rb') as fd:
            self.x_test = pickle.load(fd)
        with open(os.path.join(input_path, 'dataPreprocess_y_test.pkl'), 'rb') as fd:
            self.y_test = pickle.load(fd)
        with open(model_path, 'rb') as fd:
            self.model = pickle.load(fd)
        print("Input the x_test, y_test and model.")
        return self.x_test, self.y_test, self.model
    
    def prediction(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

    def metricsAcc(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        a = cm.shape
        corrPred = 0
        falsePred = 0

        for row in range(a[0]):
            for c in range(a[1]):
                if row == c:
                    corrPred +=cm[row,c]
                else:
                    falsePred += cm[row,c]
        self.acc = corrPred/(cm.sum())
        return self.acc

    def loadTo(self,output_path):
        with open(output_path, 'w') as fd:
            fd.write('{:4f}\n'.format(self.acc))
        print("Output the Accuracy.")


if __name__ == '__main__':
    input_data = sys.argv[1]
    input_model = sys.argv[2]
    output = sys.argv[3]

    evaluate = evaluation()
    evaluate.x_test, evaluate.y_test, evaluate.model = evaluate.loadData(input_path = input_data,model_path=input_model)
    evaluate.y_pred = evaluate.prediction()
    evaluate.auc = evaluate.metricsAcc()
    evaluate.loadTo(output_path = output)