import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)

sepal_length=iris['data'][:,0]
pedal_length=iris['data'][:,2]

c=iris['target']

plt.scatter(sepal_length,pedal_length,c=c)
plt.xlabel('Sepal Length')
plt.ylabel('Pedal Length')
plt.show()

X=np.column_stack((sepal_length,pedal_length))
y=iris.target 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

clf = svm.SVC(kernel='poly',degree=3,C=1)

clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

plot_decision_regions(X=X_test,y=y_test,clf=clf,legend=1)


