import pandas as pd
import numpy as np
from sklearn import datasets

# importing datasets
mnist=datasets.load_digits()
# assigning feature to x and label to y
x=mnist.data
y=mnist.target

# importing knn classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
# importing train_test_split data into trainig and testing from sklearn
from  sklearn.model_selection import train_test_split

# distributing the dataset intotraining and testing and test size 20% for testing the model om prediction
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=1,stratify=y)
knn=KNeighborsClassifier()

# fitting or training the model
knn.fit(x_train,y_train)
#  predicting the model 
y_pred=knn.predict(x_test)
# testing the accuracy of prediction
print(knn.score(x_test,y_test))

# 0.9888888888888889
# accuracy of model on prediction
