import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

dataset = pd.read_csv('Training.csv')

X = dataset.iloc[:, :132].values
y = dataset.iloc[:, -1].values


labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
dump(model, 'model.pkl')
