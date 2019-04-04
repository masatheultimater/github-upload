from sklearn import datasets
from sklearn import svm

#Irisの測定データ読み込み
iris = datasets.load_iris()
print(iris.data)
print(iris.data.shape)

num = len(iris.data)
print(num)

clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)
