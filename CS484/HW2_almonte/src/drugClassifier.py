import numpy as np
import re
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import cross_validate, train_test_split # Import train_test_split function
from sklearn import feature_selection, metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB


train_matrix = np.zeros((800,100001)) # create zero filled train_matrix
test_matrix = np.zeros((350,100001))

train_file = open('../train_drugs.txt','r')
test_file = open('../test.txt','r')


    
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


# Build Decision Tree Classifier and fit using binary train_matrix and train_matrix_labels
dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_matrix,train_matrix_labels)

# Feature selection is based on the Gini importance
importance = dtc.feature_importances_
important_features = {}
for i,v in enumerate(importance):
    if v>0:
        important_features[i] = v

# Deteles features with 0 importance from both testing and training sets 
train_matrix = np.delete(train_matrix, [x for x in range(1000001) if(x not in important_features) ],1)
test_matrix = np.delete(test_matrix, [x for x in range(1000001) if(x not in important_features) ],1)

# Training data set is split 1:4 ratio of testing to training
x_train, x_test, y_train,y_test= train_test_split(train_matrix,train_matrix_labels,random_state=8,test_size=0.2)

# Computes an dictionary consisting keys [ccp_alpha,impurities] each with a list of values 
path = dtc.cost_complexity_pruning_path(x_train,y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

best_ccp_alpha = 0.0
best_score= 0.0

# Builds trees using a cpp_alpha value and tests against the testing file matrix using F1 scores
for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtc.fit(x_train,y_train)
    train_prediction = dtc.predict(x_train)
    y_test = list(map(int,y_test))
    test_data_cross_val = cross_val_score(dtc,x_test,y_test,cv=10,scoring='f1')
    test_data_cross_val = list(map(float,test_data_cross_val))
    

    if np.average(test_data_cross_val) > best_score and ccp_alpha > 0.0:
        best_ccp_alpha = ccp_alpha
        best_score = np.average(test_data_cross_val)

        

print('Best Cross-Validation Score {} with alpha {}'.format(best_score, best_ccp_alpha))

# Build a new Tree and record predictions onto a file
dtc = DecisionTreeClassifier(ccp_alpha= best_ccp_alpha)
dtc.fit(train_matrix,train_matrix_labels)
predictions = dtc.predict(test_matrix)
predictions_file = open('../predictions_file.txt','w')
for i in predictions:
    predictions_file.writelines(i)
    predictions_file.writelines('\n')
    

######### Naive Bayes Classifier
nbc = BernoulliNB() # Initializing Classifier
bayes_X_train, bayes_X_test, bayes_y_train, bayes_y_test = train_test_split(train_matrix, train_matrix_labels, test_size=0.2, random_state=0)
nbc.fit(bayes_X_train,bayes_y_train)
nbc_pred = nbc.predict(bayes_X_test)

bayes_y_test = list(map(int,bayes_y_test))
print('bayes classifier: ',np.average(cross_val_score(BernoulliNB(),bayes_X_test,bayes_y_test,cv=10,scoring='f1')))

# print(nbc.feature_log_prob_)

# predictions_file.close()