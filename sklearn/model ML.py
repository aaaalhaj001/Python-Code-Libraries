#___________________________________________________________#
""" Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.4)
logre.fit(x_train,y_train)
y_p=logre.predict(x_test)
print (metrics.accuracy_score(y_test,y_p))


KNN
from sklearn.neighbors import KNeighborsClassifier  
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.4)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pm=knn.predict(x_test)
print(metrics.accuracy_score(y_test,pm)) """

k=0
while K==30:
 pipeline_knn=Pipeline([('scalar4',StandardScaler()), ('pca4',PCA(n_components=6)), ('knn_Regressor',KNeighborsRegressor(n_neighbors=k,leaf_size=1,p=1))])
 k=k+1
 pipeline_knn.fit(x_train,y_train)
 #predict x_test values
 pred=pipeline_knn.predict(x_test)
#print accuracy for algorithm
 print("Accuracy for KNeighbors  data: ",knn.score(x_test,y_test))