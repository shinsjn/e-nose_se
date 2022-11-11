import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

temp = np.loadtxt('concatdata.dat')
x = temp[:,:-1]
y = temp[:,-1]

x_untransformed = x
x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=33,stratify=y)


XGB = XGBClassifier(eval_metric='mlogloss',use_label_encoder=False,objective='mulit:sofprob')
SVC = SVC(kernel='poly')
LR = LogisticRegression()
RF = RandomForestClassifier()
DT = DecisionTreeClassifier()
KNN = KNeighborsClassifier()
LGB = LGBMClassifier()
GB = GradientBoostingClassifier()


lda = LDA(n_components=2)
pca = PCA(n_components=2)
LDA_reduced = lda.fit_transform(x,y)
PCA_reduced = pca.fit_transform(x)

'''
## 시각화
df=pd.DataFrame(np.column_stack([LDA_reduced,y]), columns =["X1","class"])

plt.scatter(df.loc[df["class"] == 0]["X1"],df.loc[df["class"] == 0]["X2"], color = 'r', label="NO2")
plt.scatter(df.loc[df["class"] == 1]["X1"],df.loc[df["class"] == 1]["X2"], color = 'g', label="NH3")
plt.scatter(df.loc[df["class"] == 2]["X1"],df.loc[df["class"] == 2]["X2"], color = 'b', label ="1:1")
plt.scatter(df.loc[df["class"] == 3]["X1"],df.loc[df["class"] == 3]["X2"], color = 'black', label="1:2")
plt.scatter(df.loc[df["class"] == 4]["X1"],df.loc[df["class"] == 4]["X2"], color = 'limegreen', label="1:3")
plt.scatter(df.loc[df["class"] == 5]["X1"],df.loc[df["class"] == 5]["X2"], color = 'violet', label ="1:4")
plt.scatter(df.loc[df["class"] == 6]["X1"],df.loc[df["class"] == 6]["X2"], color = 'dogerblue', label="1:5")
plt.legend()
plt.show()
'''


'''
#parameter tuning
#=============================XGB==============================
param_grid_XGB = [
    {
    'booster':['gbtree','dart'],
    'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],
     'max_depth':[4,5,6,7,8,9],
     'colsample_bytree':[0.5,0.8,0.9,1],
     'reg_lambda':[0,0.8,0.9,1],
     'reg_alpha':[0,0.1,0.3,0.5,1,2,3],
     'gamma':[0,0.3,0.5,1,2],
        'min_child_weight':[1,2,3],
        'n_estimators':[100,200,300]
    }
  ]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',XGB)])
random_search = RandomizedSearchCV(XGB, param_grid_XGB, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10,n_iter=150,random_state=33)
random_search.fit(x,y)
scores = pd.DataFrame(random_search.cv_results_)
scores.to_csv('./XGB.csv')
'''
#=============================LR==============================
param_grid_LR = [
    {
    'penalty':['l1','l2'],
     'C':[1e-2,1e-1,1,5,10],
     'max_iter':[1000,2000,3000],
    'solver':['liblinear']
    }
  ]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',LR)])

grid_search = GridSearchCV(LR, param_grid_LR, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)
scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LR.csv')
'''
#=============================GB==============================
params_GB = {
    'n_estimators':[50,100,200,400],
     'learning_rate':[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6]
         }
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',GB)])

grid_search = GridSearchCV(GB, params_GB, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./GB.csv')

#=============================DT==============================
parameters_DT = {'max_depth': [3, 5, 7,10],
              'min_samples_split': [3, 5,7,10],
              'splitter': ['best', 'random'],
              'criterion': ['gini','entropy'],
              'max_features':['int', 'float', 'auto', 'sqrt', 'log2','None']}
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',DT)])

grid_search = GridSearchCV(DT, parameters_DT, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./DT.csv')


#=============================SVM==============================

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid_Logistic_svc = [{'classify__C': param_range,
               'classify__kernel': ['linear']},
              {'classify__C': param_range,
               'classify__gamma': param_range,
               'classify__kernel': ['rbf']}]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',SVC)])

grid_search = GridSearchCV(SVC, param_grid_Logistic_svc, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)
print("1")
scores = pd.DataFrame(grid_search.cv_results_)
print("2")
scores.to_csv('./SVM.csv')
print("3")

#=============================KNN==============================

param_grid_KNN = [
    {
    'n_neighbors' : list(range(1,50)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski']
    }
  ]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',KNN)])

grid_search = GridSearchCV(KNN, param_grid_KNN, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./KNN.csv')

#=============================LGBM==============================

parameters = {'num_leaves':[20,40,60,80,100], 'min_child_samples':[5,10,15],'max_depth':[-1,5,10,20],
             'learning_rate':[0.05,0.1,0.2],'reg_alpha':[0,0.01,0.03]}
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',LGB)])

grid_search = GridSearchCV(LGB, parameters, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LGBM.csv')

#=============================RF==============================


params_RF = { 'n_estimators' : [100,200,400],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [10,20,30],
           'min_samples_split' : [10,20,30]
            }
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',RF)])

grid_search = GridSearchCV(RF, params_RF, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./RF.csv')
'''

'''
#####################################################################PCA

#=============================XGB==============================
param_grid_XGB = [
    {
    'classify__booster':['gbtree'],
    'classify__learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],
     'classify__max_depth':[4,5,6,7,8,9],
     'classify__colsample_bytree':[0.5,0.8,0.9,1],
     'classify__reg_lambda':[0,0.8,0.9,1],
     'classify__reg_alpha':[0,0.1,0.3,0.5,1,2,3],
     'classify__gamma':[0,0.3,0.5,1,2],
        'classify__min_child_weight':[1,2,3],
        'classify__n_estimators':[100,200,300]
    }
  ]
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',XGB)])
random_search = RandomizedSearchCV(pipe, param_grid_XGB, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10,n_iter=150,random_state=33)
random_search.fit(x,y)
scores = pd.DataFrame(random_search.cv_results_)
scores.to_csv('./PCA_XGB.csv')

#=============================LR==============================
param_grid_LR = [
    {
    'classify__penalty':['l1','l2'],
     'classify__C':[1e-2,1e-1,1,5,10],
     'classify__max_iter':[1000],
    }
  ]
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',LR)])

grid_search = GridSearchCV(pipe, param_grid_LR, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)
scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./PCA_LR.csv')

#=============================GB==============================
params_GB = {
    'classify__n_estimators':[50,100,200,400],
     'classify__learning_rate':[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6]
         }
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',GB)])

grid_search = GridSearchCV(pipe, params_GB, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./PCA_GB.csv')

#=============================DT==============================
parameters_DT = {'classify__max_depth': [3, 5, 7,10],
              'classify__min_samples_split': [3, 5,7,10],
              'classify__splitter': ['best', 'random'],
              'classify__criterion': ['gini','entropy'],
              'classify__max_features':['int', 'float', 'auto', 'sqrt', 'log2','None']}
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',DT)])

grid_search = GridSearchCV(pipe, parameters_DT, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./PCA_DT.csv')
'''

'''
#=============================SVM==============================

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid_Logistic_svc = [{'classify__C': param_range,
               'classify__kernel': ['linear']},
              {'classify__C': param_range,
               'classify__gamma': param_range,
               'classify__kernel': ['rbf']}]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',SVC)])

grid_search = GridSearchCV(SVC, param_grid_Logistic_svc, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)
print("1")
scores = pd.DataFrame(grid_search.cv_results_)
print("2")
scores.to_csv('./SVM.csv')
print("3")
'''

'''
#=============================KNN==============================

param_grid_KNN = [
    {
    'classify__n_neighbors' : list(range(1,50)),
    'classify__weights' : ["uniform", "distance"],
    'classify__metric' : ['euclidean', 'manhattan', 'minkowski']
    }
  ]
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',KNN)])

grid_search = GridSearchCV(pipe, param_grid_KNN, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./PCA_KNN.csv')

#=============================LGBM==============================

parameters = {'classify__num_leaves':[20,40,60,80,100], 'classify__min_child_samples':[5,10,15],'classify__max_depth':[-1,5,10,20],
             'classify__learning_rate':[0.05,0.1,0.2],'classify__reg_alpha':[0,0.01,0.03]}
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',LGB)])

grid_search = GridSearchCV(pipe, parameters, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./PCA_LGBM.csv')

#=============================RF==============================


params_RF = { 'classify__n_estimators' : [100,200,400],
           'classify__max_depth' : [6, 8, 10, 12],
           'classify__min_samples_leaf' : [10,20,30],
           'classify__min_samples_split' : [10,20,30]
            }
pipe = Pipeline([('PCA', PCA(n_components=1)),('classify',RF)])

grid_search = GridSearchCV(pipe, params_RF, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./PCA_RF.csv')

###############################################################################LDA
#=============================XGB==============================
param_grid_XGB = [
    {
    'classify__booster':['gbtree'],
    'classify__learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],
     'classify__max_depth':[4,5,6,7,8,9],
     'classify__colsample_bytree':[0.5,0.8,0.9,1],
     'classify__reg_lambda':[0,0.8,0.9,1],
     'classify__reg_alpha':[0,0.1,0.3,0.5,1,2,3],
     'classify__gamma':[0,0.3,0.5,1,2],
        'classify__min_child_weight':[1,2,3],
        'classify__n_estimators':[100,200,300]
    }
  ]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',XGB)])
random_search = RandomizedSearchCV(pipe, param_grid_XGB, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10,n_iter=150,random_state=33)
random_search.fit(x,y)
scores = pd.DataFrame(random_search.cv_results_)
scores.to_csv('./LDA_XGB.csv')

#=============================LR==============================
param_grid_LR = [
    {
    'classify__penalty':['l1','l2'],
     'classify__C':[1e-2,1e-1,1,5,10],
     'classify__max_iter':[1000],
    }
  ]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',LR)])

grid_search = GridSearchCV(pipe, param_grid_LR, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)
scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LDA_LR.csv')

#=============================GB==============================
params_GB = {
    'classify__n_estimators':[50,100,200,400],
     'classify__learning_rate':[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6]
         }
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',GB)])

grid_search = GridSearchCV(pipe, params_GB, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LDA_GB.csv')

#=============================DT==============================
parameters_DT = {'classify__max_depth': [3, 5, 7,10],
              'classify__min_samples_split': [3, 5,7,10],
              'classify__splitter': ['best', 'random'],
              'classify__criterion': ['gini','entropy'],
              'classify__max_features':['int', 'float', 'auto', 'sqrt', 'log2','None']}
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',DT)])

grid_search = GridSearchCV(pipe, parameters_DT, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LDA_DT.csv')


#=============================SVM==============================

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid_Logistic_svc = [{'classify__C': param_range,
               'classify__kernel': ['linear']},
              {'classify__C': param_range,
               'classify__gamma': param_range,
               'classify__kernel': ['rbf']}]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',SVC)])

grid_search = GridSearchCV(SVC, param_grid_Logistic_svc, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)
print("1")
scores = pd.DataFrame(grid_search.cv_results_)
print("2")
scores.to_csv('./SVM.csv')
print("3")

#=============================KNN==============================

param_grid_KNN = [
    {
    'classify__n_neighbors' : list(range(1,50)),
    'classify__weights' : ["uniform", "distance"],
    'classify__metric' : ['euclidean', 'manhattan', 'minkowski']
    }
  ]
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',KNN)])

grid_search = GridSearchCV(pipe, param_grid_KNN, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LDA_KNN.csv')

#=============================LGBM==============================

parameters = {'classify__num_leaves':[20,40,60,80,100], 'classify__min_child_samples':[5,10,15],'classify__max_depth':[-1,5,10,20],
             'classify__learning_rate':[0.05,0.1,0.2],'classify__reg_alpha':[0,0.01,0.03]}
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',LGB)])

grid_search = GridSearchCV(pipe, parameters, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LDA_LGBM.csv')

#=============================RF==============================


params_RF = { 'classify__n_estimators' : [100,200,400],
           'classify__max_depth' : [6, 8, 10, 12],
           'classify__min_samples_leaf' : [10,20,30],
           'classify__min_samples_split' : [10,20,30]
            }
pipe = Pipeline([('LDA', LDA(n_components=1)),('classify',RF)])

grid_search = GridSearchCV(pipe, params_RF, cv=4,n_jobs=-1,
                           scoring='accuracy', return_train_score=True,verbose=10)
grid_search.fit(x,y)

scores = pd.DataFrame(grid_search.cv_results_)
scores.to_csv('./LDA_RF.csv')
'''


