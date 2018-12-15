# importing necessory packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

# creating all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state = 9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state = 9)))

# variable to hold results and names
results = []
names = []
scoring = "accuracy"

# importing feature vector and trained labels
h5f_data = h5py.File('C:\\Users\\himanshu\\Desktop\\flowers classification\\output\\data.h5', 'r')
h5f_label = h5py.File('C:\\Users\\himanshu\\Desktop\\flowers classification\\output\\label.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
print(global_labels)
h5f_data.close()
h5f_label.close()

# verifying the shape of the features vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started..")


# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features), np.array(global_labels), test_size = 0.10, random_state = 9)

print("[STATUS] splitted the train and test data...")
print("Train data : {}".format(trainDataGlobal.shape))
print("Test data : {}".format(testDataGlobal.shape))
print("Test label : {}".format(testLabelsGlobal.shape))
print("Train label : {}".format(trainLabelsGlobal.shape))

import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# boxplot algoritham comparison
fig = pyplot.figure()
fig.suptitle(" ML algoritham comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

import matplotlib.pyplot as plt
import sys

sys.path.append('C:/Users/himanshu/Desktop/flowers classification')
import Global
# creating random forest model
clf = RandomForestClassifier(n_estimators=100, random_state = 9)

clf.fit(trainDataGlobal, trainLabelsGlobal)

test_path = "C:\\Users\\himanshu\\Desktop\\flowers classification\\dataset\\test"

for file in glob.glob(test_path + "/*.jpg"):
    image = cv2.imread(file)
    
    image = cv2.resize(image, (500, 500))
    fv_hu_moments = Global.fd_hu_moments(image)
    fv_haralick = Global.fd_haralick(image)
    
    global_feature = np.hstack([fv_hu_moments, fv_haralick])
    
    #predicting label
    prediction = clf.predict(global_feature.reshape(1, -1))[0]
    
    #showing predicted label on image
    cv2.putText(image, Global.train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()











