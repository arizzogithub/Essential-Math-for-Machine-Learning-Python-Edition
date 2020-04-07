#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import standard libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn.metrics as metrics

from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve
)

from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    MaxAbsScaler, 
    RobustScaler, 
    QuantileTransformer, 
    PowerTransformer, 
    Normalizer
)

from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)

import warnings
warnings.filterwarnings("ignore")



def models (X_train, X_test, y_train, y_test):
    
    # Logistic Regression
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    LR_pred = LR.predict(X_test)
    LR_accuracy = accuracy_score(LR_pred, y_test)
    
    # K-Nearest Neighbours
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)
    KNN_pred = KNN.predict(X_test)
    KNN_accuracy = accuracy_score(KNN_pred, y_test)
    
    # Support Vector Machine
    SVCc = svm.SVC()
    SVCc.fit(X_train, y_train)
    SVCc_pred = SVCc.predict(X_test)
    SVCc_accuracy = accuracy_score(SVCc_pred, y_test)
    
    # Decision Tree
    DT = DecisionTreeClassifier(random_state = 0)
    DT.fit(X_train, y_train)
    DT_pred = DT.predict(X_test)
    DT_accuracy = accuracy_score(DT_pred, y_test)
    
    # Random Forest
    RF = RandomForestClassifier(random_state = 0)
    RF.fit(X_train, y_train)
    RF_pred = RF.predict(X_test)
    RF_accuracy = accuracy_score(RF_pred, y_test)
    
    # Gradient Boosting
    GB = GradientBoostingClassifier()
    GB.fit(X_train, y_train)
    GB_pred = GB.predict(X_test)
    GB_accuracy = accuracy_score(GB_pred, y_test)
    
    # Naive Bayes
    NB = GaussianNB()
    NB.fit(X_train, y_train)
    NB_pred = NB.predict(X_test)
    NB_accuracy = accuracy_score(NB_pred, y_test)


    # visualisation table
    table = {
      'Model': ['Logistic Regression', 
                'K-Nearest Neighbours', 
                'Support Vector Machine',
                'Decision Tree', 
                'Random Forest', 
                'Gradient Boosting', 
                'Naive Bayes'],
        
      'Accuracy without hyper parameters tuned': [LR_accuracy, 
                                                  KNN_accuracy, 
                                                  SVCc_accuracy,
                                                  DT_accuracy, 
                                                  RF_accuracy, 
                                                  GB_accuracy, 
                                                  NB_accuracy]}

    final_table = pd.DataFrame(data = table)
    return final_table


        # visualisation plot
    x = ['Naive Bayes', 'Gradient Boosting', 'Random Forest', 'Decision Tree',
         'Support Vector Machine', 'K-Nearest Neighbours', 'Logistic Regression']
    y = [NB_accuracy, GB_accuracy, RF_accuracy,  DT_accuracy, SVCc_accuracy, KNN_accuracy, LR_accuracy,]
    colors = ["mediumseagreen", "lightblue", "royalblue", "mediumseagreen",
              "lightblue", "royalblue", "mediumseagreen"]
    fig, ax = plt.subplots()
    plt.barh(y = range(len(x)), tick_label = x, width = y, height = 0.35, color = colors);
    ax.set(xlabel = 'Accuracy without hyper parameters tuned', ylabel = 'Model');
    print (plt)

   





def knn_with_hyper_param (X_train, X_test, y_train, y_test):

    KNN = KNeighborsClassifier()
    KNN_params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
    KNN1 = GridSearchCV(KNN, param_grid = KNN_params)
    KNN1.fit(X_train, y_train)
    print("K-Nearest Neighbour Best Hyper Parameters:   ", KNN1.best_params_)
    KNN1_pred = KNN1.predict(X_test)
    KNN1_accuracy = accuracy_score(KNN1_pred, y_test)
    print('KNN accuracy (with hyper parameters):    ', KNN1_accuracy)

