import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler

#Data can be downloaded from https://www.kaggle.com/ruslankl/mice-protein-expression
DATA_PATH = 'Data_Cortex_Nuclear.csv'


def load_data(data_path=DATA_PATH):
    return pd.read_csv(data_path)


def main():
    # Load our data
    ctx_data = load_data()
    
    # Replace NaN values with 0
    ctx_data.replace(np.nan, 0, inplace=True)

    # Split data into features and labels 
    X = ctx_data[ctx_data.columns.values.tolist()[1:-4]]
    Y = ctx_data[ctx_data.columns.values.tolist()[-4:]]

    # Center the data
    X_std = StandardScaler().fit_transform(X)

    # Features are columns from X_std
    features = X_std.T

    # create a covariance matrix from the features
    covariance_matrix = np.cov(features)

    # Calculate eigen values and vectors from the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Find the minimum amount of components needed 
    # To retain the majority of the information (80% in this case)
    pca_count = 0
    s = 0
    eigen_sum = sum(eigen_values)

    for i in range(len(eigen_values)):
        s += eigen_values[i]
        if s/eigen_sum > 0.80:
            pca_count = i+1
            break

    projected_X = np.array([X_std.dot(eigen_vectors.T[i]) for i in range(pca_count+1)])

    # Create a new data frame for all the PCs
    result = pd.DataFrame(projected_X.T, columns=['PC%s' % str(i+1) for i in range(pca_count+1)])
    result['class'] = Y['class']

#    print(result.head())
    g = sns.lmplot('PC1', 'PC2', data=result, fit_reg=False, legend_out=True, scatter_kws={'s': 40},
            hue='class')

    plt.title('PCA')

    plt.show()


if __name__ == '__main__':
    main()
