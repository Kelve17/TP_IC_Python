# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 02:08:22 2018

@author: Kelve Neto
"""

# Import modules

import numpy as np
import numpy.fft as fft
import scipy as sc
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import math as mt
from statsmodels import robust
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
from statistics import median
from statistics import mode
from scipy.stats import iqr
from sklearn.model_selection import train_test_split
# Import PySwarms
import pyswarms as ps
from iteration_utilities import deepflatten
from sklearn.svm import SVC
from sklearn import metrics,linear_model


data = np.loadtxt("features.csv",delimiter=",");
X = data[:,1:];
lr = np.arange(22);
y = data[:,0].astype(int)
#y = (lr+1==y[:,None]).astype(int)


##dividing the dataset into training and testing (training 60% and test=40%)
#X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.3)
#
## Plot toy dataset per feature
##
##df = pd.DataFrame(X)
##
##df['labels'] = pd.Series(y)
##
##
##
##sns.pairplot(df, hue='labels');
##
##
##

##
##
##
### Create an instance of the classifier
##
#classifier = SVC(kernel="linear", C=0.015)
classifier = SVC(gamma=2, C=1)
#classifier.fit(X,y)
#y_pred = classifier.predict(X)
#cm = confusion_matrix(y,y_pred)
#fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
#auc = "%.2f" % metrics.auc(fpr, tpr)
#print(cm)  
#print(classification_report(y,y_pred))
#print("f_measure : ", f1_score(y, y_pred, average="macro"))
#print("precision : ", precision_score(y, y_pred, average="macro"))
#print("recall : ",recall_score(y, y_pred, average="macro")) 
#print("auc : ", auc)  

##
##
### Define objective function
##
def f_per_particle(m, alpha):
##
    """Computes for the objective function per particle



   Inputs

    ------

    m : numpy.ndarray

        Binary mask that can be obtained from BinaryPSO, will

        be used to mask features.

    alpha: float (default is 0.5)

        Constant weight for trading-off classifier performance

        and number of features



    Returns

    -------

    numpy.ndarray

        Computed objective function

    """

    total_features = 161
##
    # Get the subset of the features from the binary mask

    if np.count_nonzero(m) == 0:

        X_subset = X

    else:

        X_subset = X[:,m==1]

    # Perform classification and store performance in P

    classifier.fit(X_subset, y)

    P = (classifier.predict(X_subset) == y).mean()

    # Compute for the objective function

    j = (alpha * (1.0 - P)

        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))



    return j



def f(x, alpha=0.88):

    """Higher-level method to do classification in the

    whole swarm.



    Inputs

    ------

   x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search



    Returns

    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle

   """

    n_particles = x.shape[0]

    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]

    return np.array(j)



# Initialize swarm, arbitrary

options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}



# Call instance of PSO

dimensions = 161 # dimensions should be the number of features

#optimizer.reset()

optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)



# Perform optimization

cost, pos = optimizer.optimize(f, print_step=1, iters=5, verbose=2)

# Create two instances of LogisticRegression

classifier = SVC(gamma=20, C=1)



# Get the selected features from the final positions

X_selected_features = X[:,pos==1]  # subset



# Perform classification and store performance in P

classifier.fit(X_selected_features, y)



# Compute performance

subset_performance = (classifier.predict(X_selected_features) == y).mean()





print('Subset performance: %.3f' % (subset_performance))



# Plot toy dataset per feature

#df1 = pd.DataFrame(X_selected_features)

#df1['labels'] = pd.Series(y)



#sns.pairplot(df1, hue='labels')
#
out = classifier.predict(X_selected_features)

#Y = (lr+1==Y[:,None]).astype(int)
cm = confusion_matrix(y,out);
fpr, tpr, thresholds = metrics.roc_curve(y, out, pos_label=1)
auc = "%.2f" % metrics.auc(fpr, tpr)
acc = accuracy_score(y,out);
print(cm)  
print(classification_report(y,out))
print("f_measure : ", f1_score(y, out, average="macro"))
print("precision : ", precision_score(y, out, average="macro"))
print("recall : ",recall_score(y, out, average="macro")) 
print("auc : ", auc)  
print("accuracy : ",acc)
