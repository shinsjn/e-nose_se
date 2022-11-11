from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/shins/Desktop/MLPA/E-nose/code/shin_prof_code/enose_codes/codes/concat_pd_data.csv')
xgb = XGBClassifier()

train_x = df.loc[:, df.columns!= 'label']
train_y = df['label']





'''
model = XGBClassifier(random_state=11)
model.fit(train_x, train_y)

# 배열형태로 반환
ft_importance_values = model.feature_importances_

# 정렬과 시각화를 쉽게 하기 위해 series 전환
ft_series = pd.Series(ft_importance_values, index = train_x.columns)
ft_top20 = ft_series.sort_values(ascending=False)[:]

# 시각화
plt.figure(figsize=(8,6))
plt.title('Feature Importance')
sns.barplot(x=ft_top20, y=ft_top20.index)
plt.show()
'''

'''
from lightgbm import plot_importance,LGBMClassifier

lgbm_wrapper = LGBMClassifier()
lgbm_wrapper.fit(train_x,train_y,verbose=True)

f, ax = plt.subplots(figsize=(6,6))
plot_importance(lgbm_wrapper, ax=ax)
plt.show()
'''
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500, random_state=1234)
# Train the model using the training sets

#model.fit(train_X, train_y)
model.fit(train_x,train_y)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

for f in range(train_x.shape[1]):
    print("{}. feature {} ({:.3f})".format(f + 1,train_x.columns[indices][f], importances[indices[f]]))

plt.figure()
plt.title('importance')
plt.bar(range(train_x.shape[1]), importances[indices],color="r", yerr = std[indices], align='center')
plt.xticks(range(train_x.shape[1]),train_x.columns[indices], rotation=45)
plt.xlim([-1,train_x.shape[1]])
plt.show()