import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
import joblib
iris = datasets.load_iris(as_frame=True)
iris = iris.frame

x = iris.drop('target', axis=1)
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=True)
lr = LogisticRegression()
lr.fit(x_train,y_train)

joblib.dump(lr, "model.pkl")
print('Model dumped!')

lr = joblib.load('model.pkl')

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print('Model columns dumped!')