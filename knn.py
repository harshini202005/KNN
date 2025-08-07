import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris=load_iris()

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
df['target_name']=df['target'].apply(lambda x:iris.target_names[x])

print("Iris dataset sample:\n")
print(df.head())

X=df[iris.feature_names]
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)   #K=3
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print("\nModel Evaluation:\n")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(knn, f)