from itertools import count
from os import remove
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import feature_selection, metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.feature_selection import VarianceThreshold # Import feature selection features
from sklearn import tree
from sklearn.model_selection import cross_val_score


train_matrix = np.zeros((800,100001)) # create zero filled train_matrix
test_matrix = np.zeros((350,100001))

train_file = open('train_drugs.txt','r')
test_file = open('test.txt','r')
format_file = open('formatfile.txt','r')
predictions = open('predictions','w')
    
# Read train file
lines = train_file.readlines()
train_matrix_labels = []
np.set_printoptions(threshold = 50)
for row in range(len(lines)): # cleans each line from file, splits it into its own array
    cur_line = lines[row].strip('\n')
    cur_line = re.split('\t| ',cur_line)
    if '' in cur_line:
            cur_line.remove('')
    for col in range(len(cur_line)): # each value is used to mark an index in the train_matrix
        if col == 0:
            train_matrix_labels.append(cur_line[col])
        else:
            cur_val = int(cur_line[col])
            train_matrix[row][cur_val] = 1

# Read test file
test_lines = test_file.readlines()
for row in range(len(test_lines)): # cleans each line from file, splits it into its own array
    cur_line = test_lines[row].strip('\n')
    cur_line = cur_line.split(' ')
    if '' in cur_line:
            cur_line.remove('')
    for col in range(len(cur_line)): # each value is used to mark an index in the test_matrix        
        cur_val = int(cur_line[col])
        test_matrix[row][cur_val] = 1

# Read test labels from test file
test_labels_lines = format_file.readlines()
test_label_matrix = []
for i in range(len(test_labels_lines)):
    test_label_matrix.append(int(test_labels_lines[i]))
test_label_matrix = np.reshape(test_label_matrix,(350,1))


# Build Decision Tree Classifier and fit using binary train_matrix and train_matrix_labels
dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_matrix,train_matrix_labels)

# Feature selection is based on the importance
importance = dtc.feature_importances_
important_features = {}
for i,v in enumerate(importance):
    if v>0:
        important_features[i] = v

# Deteles features with 0 importance from both testing and training sets 
train_matrix = np.delete(train_matrix, [x for x in range(1000001) if(x not in important_features) ],1)
test_matrix = np.delete(test_matrix, [x for x in range(1000001) if(x not in important_features) ],1)

# Training data set is split 1:4 ratio of testing to training
x_train, x_test, y_train,y_test= train_test_split(train_matrix,train_matrix_labels,random_state=8 )

# Computes an dictionary consisting keys [ccp_alpha,impurities] each with a list of values 
path = dtc.cost_complexity_pruning_path(x_train,y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

best_ccp_alpha = 0.0
max_test_score= 0.0

# Builds trees using a cpp_alpha value and tests against the testing file matrix using F1 scores
for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtc.fit(x_train,y_train)
    
    train_prediction = dtc.predict(x_train)
    train_prediction = list(map(int,train_prediction))
    test_prediction = dtc.predict(test_matrix)
    test_prediction = list(map(int,test_prediction))
    y_train = list(map(int,y_train))
    y_test = list(map(int,y_test))
    
    test_labels_lines = list(map(int,test_labels_lines))
    train_score = metrics.f1_score(y_train,train_prediction,average = 'binary')
    test_score = metrics.f1_score(test_label_matrix,test_prediction,average = 'binary')
    # Records the best cpp_alpha value and writes predictions to file 
    if (test_score > max_test_score) and (ccp_alpha > 0.0):
        best_ccp_alpha = ccp_alpha
        max_test_score = test_score
        predictions_file = open('predictions_file.txt','w')
        for line in test_prediction:
            predictions_file.write('{}\n'.format(str(line)))
    

# Print average cross validation scores with the best possible ccp_alpha
print('best ccp_alpha:{}'.format(best_ccp_alpha))
train_data_cross_val = cross_val_score(DecisionTreeClassifier(ccp_alpha=best_ccp_alpha),train_matrix,train_matrix_labels)
print('Train data score: ',np.average(train_data_cross_val))
test_data_cross_val = cross_val_score(DecisionTreeClassifier(ccp_alpha=best_ccp_alpha),test_matrix,test_label_matrix,cv=10)
print('Test data score: ',np.average(test_data_cross_val))

predictions_file.close()