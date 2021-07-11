#!/usr/bin/env python
# coding: utf-8

# # Prediction using Unsupervised ML
# 
# ● From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# 
# ● Dataset : https://bit.ly/3kXTdox

# In[1]:


#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() # See the first 5 rows


# In[2]:


x=iris_df.iloc[:,[0,1,2,3]].values  

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
wcss

#Elbow Method- to find value of k
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')  # Within cluster sum of squares   #wcss is low for higher no. of clusters
plt.show()


# In[3]:


#Clustering
kmeans=KMeans(3)
kmeans.fit(x) #shows the parameters


# In[4]:


#results
identified_clusters=kmeans.fit_predict(x) 
identified_clusters #gives array containing predicted clusters


# In[5]:


# Visualising the clusters - On the first two columns
plt.scatter(x[identified_clusters  == 0, 0], x[identified_clusters  == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[identified_clusters  == 1, 0], x[identified_clusters  == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[identified_clusters  == 2, 0], x[identified_clusters  == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




