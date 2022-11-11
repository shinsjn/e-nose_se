import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
def print_report(labels, true, predicted):
    class_report = classification_report(true, predicted, target_names=labels)
    conf_matrix = confusion_matrix(true, predicted)
    print('\nClass Report: \n', class_report)
    plt.figure(2)
    ax = plt.subplot()
    df_cm = pd.DataFrame(conf_matrix, index=labels,columns=labels)
    # sn.set(font_scale=1.0)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

temp = np.loadtxt('concatdata.dat')
x = temp[:,:-1]
y = temp[:,-1]
print(temp.shape)
x_untransformed = x
x = StandardScaler().fit_transform(x)

clf = XGBClassifier(reg_lambda= 0, reg_alpha= 0.1, n_estimators= 100, min_child_weight= 1, max_depth= 8, learning_rate= 0.35, gamma= 1, colsample_bytree= 0.9, booster= 'dart')
# clf = SVC(kernel='poly')
# clf = LogisticRegression()
# clf = RandomForestClassifier()
# clf = DecisionTreeClassifier()
#clf = KNeighborsClassifier()
verbose = True
labels = ['NO2','NH3','1:1','1:2', '1:3', '1:4','1:5']
principal_components = 2
avg_list = []
for i in range(10):
    wrongs_x = []
    wrongs_x2 = []
    wrongs_y = []
    true2 = []
    pred2 = []
    x2 = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        k_folds = 4
    
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        true = []
        predicted = []
        scores = []
        for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)): 
            # tmp_ids = train_ids
            # train_ids = test_ids
            # test_ids = tmp_ids
            
            x_train = x[train_ids]
            x_test = x[test_ids]
            x2 = x[test_ids]    #원본
            x3 = x_untransformed[test_ids]  #원본?
            y_train = y[train_ids]
            y_test = y[test_ids]
            '''
            pca = PCA(n_components=2)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)
            print(x_train.shape)
            '''
            clf.fit(x_train, y_train)
            if (verbose):
                print('Fold ', fold, ': ', clf.score(x_test, y_test)*100, '%\n', end='')
        
            scores.append(clf.score(x_test, y_test))
            true += list(y_test)
            true2 = list(y_test)
            predicted += list(clf.predict(x_test))
            pred2 = list(clf.predict(x_test))
            for i in range(len(true2)):
                if (true2[i] != pred2[i]):
                    wrongs_x.append(x2[i])
                    wrongs_x2.append(x3[i])
                    wrongs_y.append(true2[i])
            # print('pred label:', clf.predict(x_test))
        fold_avg = np.average(scores)*100
        avg_list.append(fold_avg)
        wrongs_x = np.array(wrongs_x)
        wrongs_x2 = np.array(wrongs_x2)
        if (verbose):
            print("Average of the folds:", fold_avg)
            print_report(labels, true, predicted)

    #plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    #plt.xlabel("Principal Component1")
    #plt.ylabel("Principal Component2")
    #plt.title('PCA with %d principal components' % (principal_components))
    #plt.show()
print("Avg: %.2f +- %.2f" % (np.average(avg_list), np.std(avg_list)))

yname = []
for i in y:
    if i == 0:
        yname.append('NH3')
    elif i == 1:
        yname.append('NO2')
    elif i == 2:
        yname.append('Mixture1')
    elif i == 3:
        yname.append('Mixture2')

y2 = wrongs_x.shape[0] * ['Incorrect']

# x_pca = pca.transform(x)
# x_wrong = pca.transform(wrongs_x)
'''
x_pca = x
x_wrong = wrongs_x
correctornot = []
for i in x_pca:
    if i in x_wrong:
        correctornot.append('Incorrect')
    else:
        correctornot.append('correct')
df = pd.DataFrame()
df["y"] = yname
df['pca-one'] = x_pca[:, 0]
df['pca-two'] = x_pca[:, 1]
df['correct'] = correctornot

# df = df.append(df2)
lbs = ['1:2', '1:3', '1:4','1:5', 'Incorrect']
sn.scatterplot(x="pca-one", y="pca-two", hue=df.y.tolist(), style='correct',
                palette=sn.color_palette("hls", 5),
                data=df).set(title="PCA")
plt.show()
'''
'''
wrongs_yname = []
for i in wrongs_y:
    if i == 0:
        wrongs_yname.append('NH3')
    elif i == 1:
        wrongs_yname.append('NO2')
    elif i == 2:
        wrongs_yname.append('Mixture1')
    elif i == 3:
        wrongs_yname.append('Mixture2')
#for i, wrong in enumerate(wrongs_x2):
#    plt.plot(list(range(21141)), wrong[21141:])
#    plt.xlabel("Time")
#    plt.ylabel("Resistance")
#    plt.title(wrongs_yname[i])
#    plt.show()
'''