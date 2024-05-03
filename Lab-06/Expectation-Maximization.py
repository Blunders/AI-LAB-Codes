
\\ Expectation Maximization

import pandas as pd
import numpy as np
from scipy.special import comb
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Part 1: Data Preprocessing and Initializations

# Loading data from CSV files
df_ten_coins = pd.read_csv('/content/2020_ten_bent_coins.csv').transpose()
df_em_clustering = pd.read_csv("2020_em_clustering.csv", sep=',', header=None).transpose()

# Constants and Initializations
np.random.seed(0)
num_iterations = 100
eps = 0.01
improvement = float('inf')
epoch = 0

# Part 2: Maximum Likelihood Estimation (MLE)

# Counting number of heads and tails
heads = df_ten_coins.sum().to_numpy()
tails = 100 - heads

# Selecting coins randomly
selected_coin = np.random.randint(0, 10, size=(500,))
_, count_selected_coin = np.unique(selected_coin, return_counts=True)

# Initializing MLE vector
MLE_vector = np.zeros(10)

# Computing MLE vector
for i, j in zip(heads, selected_coin):
    MLE_vector[j] += i

MLE_vector = MLE_vector / (count_selected_coin * 100)

# Part 3: Expectation-Maximization (EM) Algorithm

# Initializing probability of heads for each coin
p_heads = np.random.random((1, 10))

# EM Algorithm Iterations
while improvement > eps and epoch < num_iterations:
    expectation = np.zeros((10, 500, 2))
    
    for i in range(500):
        e_head = heads[i]
        e_tail = tails[i]
        likelihood = np.zeros(10)
        
        # Computing likelihood for each coin
        for j in range(10):
            likelihood[j] = compute_likelihood(e_head, 100, p_heads[epoch][j])
        
        # Calculating weights based on likelihood
        weights = likelihood / np.sum(likelihood)
        
        # Expectation step: updating expectations
        for j in range(10):
            expectation[j][i] = weights[j] * np.array([e_head, e_tail])
        
        # Maximization step: updating parameters (MLE estimates)
        theta = np.zeros(10)
        for i in range(10):
            theta[i] = np.sum(expectation[i], axis=0)[0] / np.sum(expectation[i])
        p_heads[epoch + 1] = theta
    
    print(f'Epoch ->{epoch}\n Theta ->{theta}')
    
    # Checking improvement for convergence
    improvement = np.max(np.abs(p_heads[epoch + 1] - p_heads[epoch]))
    epoch += 1

# Output: MLE Estimates
for i, j in enumerate(theta):
    print(f"{i+1} : {j:.3f}")

# Part 4: K-Means Clustering

# Initializing KMeans model
kmeans = KMeans(n_clusters=2)

# Fitting and predicting clusters
kmeans.fit(df_em_clustering)
predictions_kmeans = kmeans.predict(df_em_clustering)

# Plotting KMeans clusters
plt.scatter(df_em_clustering.iloc[:, 0], [i for i in range(df_em_clustering.shape[0])], c=predictions_kmeans)
plt.xlabel("position")
plt.ylabel("Classification")
plt.show()

# Part 5: Gaussian Mixture Model (EM Clustering)

# Initializing Gaussian Mixture model
em = GaussianMixture(n_components=2)

# Fitting and predicting clusters
em.fit(df_em_clustering)
predictions_em = em.predict(df_em_clustering)

# Plotting EM clusters
plt.scatter(df_em_clustering.iloc[:, 0], [i for i in range(df_em_clustering.shape[0])], c=predictions_em)
plt.xlabel("position")
plt.ylabel("Classification")
plt.show()

