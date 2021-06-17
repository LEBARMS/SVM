import matplotlib.pyplot as plt  # Pour visualiser la base de donne graphiquement.
import numpy as np
import pandas as pd  # Pour visualiser la base de donne.
from sklearn import datasets  # Pour recuperer le dataset.
from sklearn import svm  # La librairie de l'algorithme.
from sklearn.model_selection import train_test_split  # Pour diviser notre dataset en training set et testing set.import
from sklearn.svm import SVC
# from mlxtend.plotting import plot_decision_regions  # Pour mieux visualiser les differences entre les reponses du svm.


#VALEUR:
entreex = float(input('Sepal Length?'))
print(entreex)
entreey = float(input('Petal Length?'))
print(entreey)
sepal_length = entreex
petal_length = entreey


#TABLEAU:
iris = datasets.load_iris()  # On prend le iris dataset de sklearn et on l'enregistre comme iris.

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)  # Pour visualiser le dataset en forme de tableau.
print(iris_df.head())


#INTERFACE:
sepal_length = iris['data'][:, 0]
petal_length = iris['data'][:, 2]

c = iris['target']
plt.scatter(sepal_length, petal_length, c=c)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()  # Graphe.

x = np.column_stack((sepal_length, petal_length)) # Pour unifier les longueur d'une pethale et d'un sepale en une colonne pour le svm.
y = iris.target  # Chaque valeur pour x va etre enregistrer dans y.
plt.scatter(sepal_length, petal_length)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 20% des donnes pour les tests le reste pour l'entrainement du svm.

svm = SVC(kernel='linear')  # Classification du svm.
svm.fit(x_train, y_train)

#plot_decision_regions(x_train, y_train, clf=svm, legend=1)


#PREDICTION:
#print("Testing:" + str(clf.score(x_test, y_test)))  # Pourcentage des predictions correctes quand on entraine le svm avec x_test et y_test.