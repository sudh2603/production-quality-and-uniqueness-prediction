from new_kNN import * 
import pandas as pd
import numpy as np

def accuracy(predictions, outcomes):
    return 100*np.mean(predictions==outcomes)

data =pd.read_csv("wine.csv")

#print(data.head(5))
numeric_data = data.drop("color",axis=1)

import sklearn.preprocessing as sp
scaled_data = sp.scale(numeric_data)
numeric_data =pd.DataFrame(scaled_data,columns=numeric_data.columns) 

import sklearn.decomposition as sd
pca = sd.PCA(n_components=2)
#principal_components = pca.fit_transform(numeric_data)
principal_components=pca.fit(numeric_data).transform(numeric_data)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Component 1"); plt.ylabel("Component 2")
plt.show()


predictors = np.array(numeric_data)
outcomes = np.array(data["high_quality"])


my_predictions = np.array([knn_predict(p,predictors,outcomes,k=5) for p in predictors])
percentage = accuracy(my_predictions,data.high_quality)
print(percentage)

"""
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3001)
knn.fit(predictors,outcomes)
scikit_prediction=knn.predict(predictors)
percentage = accuracy(scikit_prediction,data.high_quality)
print(percentage)
"""