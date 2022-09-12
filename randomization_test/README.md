# Randomization Test
The code used to conduct the randomization test is included in this directory. The change in prediction error associated with the use of seven regional models in comparison to one general model is evaluated using a randomization test. Under the null hypothesis, the use of regional models defined on the seven geographical regions of Turkey does not result in a reduction in prediction error in comparison with one general model for all of Turkey. Therefore, under the null hypothesis, any random assignment of stations to seven clusters has the potential of producing a similar reduction in prediction error. The alternative hypothesis states that the use of these specific regions to create regional models resulted in a reduction in prediction error.

### Randomization test steps:
* Weather stations are randomly assigned to 7 clusters, where the number of datapoints in each cluster is determined by sampling from a Dirichlet distribution, as described below, in order to obtain cluster dataset sizes similar to the dataset size in each of the seven regions used in this study.
* Stations in each cluster are split to train and test stations.
* A polynomial regression model is trained for each cluster.
* A general model is trained on all training stations combined and its prediction's RMSE is computed.
* The combined predictions of the 7 cluster models are used to compute the cluster model RMSE.
* The change in RMSE between the general and cluster models is computed using the following equation:

$$ RMSE_{change} = \frac{RMSE_{cluster} - RMSE_{general}}{RMSE_{general}} $$

* The previous steps are repeated 1000 times
* A p-value is computed by taking the ratio of iterations in which the reduction in prediction error ($RMSE_{change}$) exceeded that of the studied models.

### Method for creating cluster permutations:
1000 permutations of cluster definitions are created by randomly assigning stations to 7 clusters in each permutation. The number of datapoints in each cluster is determined by sampling from a 7th order Dirichlet distribution (Eq. 1) in each iteration. This way, the distribution of datapoints in the 7 random clusters would be similar to that of the seven regions used in this study. In other words, one cluster would simulate Central Anatolia, another cluster would simulate the Mediterranean region, and so on.

$$ f(y^7) = \frac{\Gamma(\alpha_0)}{\prod_{i=1}^{7} \Gamma(\alpha_i)} \prod_{i=1}^{7} y_i^{\alpha_i - 1} \tag{1}$$

$$  \alpha_0 = \sum^{7}_{i=1} \alpha_i $$

The parameters of the Dirichlet distribution were defined as follows:

$$ \alpha_i =  \frac{\beta n_i}{N} \tag{2}$$

Where:
* $n_i$: The number of data points in the ith region of the regions used in this study
* $N$: The total number of datapoints
* $\beta$: Concentration parameter

By using this parameter definition, the mean of this 7th order Dirichlet distribution (Eq. 3) would be a 7 dimensional vector, where each of its 7 values represents the ratio of the region dataset size to the total number of datapoints. The value of the concentration parameter determines how dispersed are the samples around the mean.

$$  E[y_i] = \frac{\alpha_i}{\alpha_0} \tag{3}$$

Source:
* Daniel Johnson (https://stats.stackexchange.com/users/8242/daniel-johnson), How to sample natural numbers, such that the sum is equal to a constant?, URL (version: 2012-03-01): https://stats.stackexchange.com/q/23959


## File description

* **combo_input_combo_test.csv**: Randomization test result for each input combination:
    * `iter`: Iteration number
    * `cluster`: Cluster number from 1 to 7. Cluster 0 represents all data combined.
    * `general_model`: RMSE of the general model trained on all the training stations combined. The RMSE is computed from using the test dataset predictions.
    * `cluster_model`: RMSE of the cluster model trained on the cluster's training stations only. The RMSE is computed from using the test dataset predictions.
    
* **cluster_permutations.csv**: Contains the 1000 permutations of cluster definitions, which were created by randomly assigning stations to 7 clusters in each permutation.

* **permutation_stats.csv**: Contains the statistics of the clusters in the 1000 permutations of the randomization test:
    * `cluster_ratio`: Dataset size ratio in each cluster relative to the entire dataset size
    * `test_ratio`: Test dataset size ratio relative to the cluster dataset size.
