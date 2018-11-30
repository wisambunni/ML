import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import metrics

#Data can be downloaded from https://www.kaggle.com/ruslankl/mice-protein-expression
DATA_PATH = 'Data_Cortex_Nuclear.csv'


def load_data(data_path=DATA_PATH):
    return pd.read_csv(data_path)


def euclidean(a, b):
    return np.linalg.norm(a-b)


def kmeans(X, k, tol=0):
    # get the number of instances and features (rows, cols)
    n_instances, n_features = X.shape 

    # define k centroids
    centroids = X[np.random.randint(0, n_instances-1, size=k)]
    # used to keep track of the old centroids at each iteration 
    centroids_old = np.zeros(centroids.shape)

    # store the clusters
    belongs_to = np.zeros((n_instances,1))
    pos = euclidean(centroids, centroids_old)

    iterations = 0
    while pos > tol:
        pos = euclidean(centroids, centroids_old)
        centroids_old = centroids
        # for each instance in the data set
        for idx_instance, instance in enumerate(X):
            # create a distance vector of size k
            dist_vec = np.zeros((k, 1))
            # for each centroid
            for idx_centroid, centroid in enumerate(centroids):
                # find the distance between point x and each centroid 
                dist_vec[idx_centroid] = euclidean(centroid, instance)

            # assign x to the cluster with the closest centroid 
            belongs_to[idx_instance, 0] = np.argmin(dist_vec)

        tmp_centroids = np.zeros((k, n_features))

        # for each k cluster 
        for idx in range(len(centroids)):
            # get all the points assigned to a cluster 
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == idx]
            # find the mean of those points
            # this will become the new centroid 
            centroid = np.mean(X[instances_close], axis=0)
            # add the new centroid to the temporary buffer 
            tmp_centroids[idx, :] = centroid 

        # update all centroids 
        centroids = tmp_centroids 

        return centroids, belongs_to


def main():
    data = load_data()
    X = data[data.columns.values.tolist()[1:-4]]
    
    # fill in empty spaces using row mean 
    X = X.apply(lambda row: row.fillna(row.mean()), axis=1)

    performance = []
    range_values = range(2, 11)
    for i in range_values:
        x = metrics.silhouette_score(X.values, kmeans(X.values, i)[1].ravel(), metric='euclidean', sample_size=len(X))
        performance.append(x)
        print(x)

    print('Optimal K amount: ',performance.index(min(performance))+2)

    plt.title('Score vs number of clusters')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.bar(range_values, performance, align='center')
    plt.show()
    

if __name__ == '__main__':
    main()
