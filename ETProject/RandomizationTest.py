import pandas as pd
import numpy as np
from scipy.stats import dirichlet
import time
import os
import yaml


# Functions for creating randomization test permutations

def get_cluster_sizes(beta, original_cluster_sizes, n_data):
    '''
    Create an array of data points per cluster using a Dirichlet distribution
    to get cluster sizes similar to the original cluster sizes.

    Source:
    https://stats.stackexchange.com/questions/23955/how-to-sample-natural-numbers-such-that-the-sum-is-equal-to-a-constant
    '''
    n_clusters = len(original_cluster_sizes)
    alpha = beta * original_cluster_sizes / n_data

    sample = dirichlet.rvs(alpha=alpha) * n_data
    sample = sample.round().astype('int')[0]

    if sample.sum() != n_data:
        diff = n_data - sample.sum()
        for _ in range(np.abs(diff)):
            i = np.random.randint(0, n_clusters)
            sample[i] += np.sign(diff)

    return sample


def split_data(stations, test_size, vals_per_station):
    '''
    Split the given stations to test and train stations iteratively
    sampling stations until the data size of test stations is approximately
    equal to test size.

    This function is used in create clusters.
    '''
    test_stations = []
    count = 0

    n_data = vals_per_station[stations].sum()
    available_stations = pd.Series(stations)

    actual_test_ratio = 1

    max_iter = 20
    i = 0

    while actual_test_ratio > 0.5 and i < max_iter:
        while count < 0.92 * test_size:
            st = available_stations.sample()
            test_stations.append(st.squeeze())

            count += vals_per_station[st.squeeze()]
            available_stations = available_stations.drop(index=st.index)

        actual_test_ratio = count / n_data
        i += 1

    return test_stations


def create_clusters(vals_per_station, vals_per_cluster, iter_num):
    '''
    Given a list of number of values per cluster and a Series of station numbers
    and the number of values in each station, this function assigns stations to clusters and split
    them to train and test stations.
    '''

    test_ratio = 0.4

    # A series of all available stations (stations are dropped from here when assigned to a cluster)
    available_stations = pd.Series(vals_per_station.index)

    df = pd.DataFrame(columns=['st_num', 'iter', 'cluster', 'dataset'])

    df_list = []

    for cluster_number, n_data in enumerate(vals_per_cluster):
        cluster_number += 1
        count = 0
        stations = []

        while count < 0.95 * n_data:
            # Randomly sampling a station
            sampled_station = available_stations.sample()
            stations.append(sampled_station.squeeze())

            count += vals_per_station[sampled_station.squeeze()]
            available_stations = available_stations.drop(index=sampled_station.index)

            # Make sure that there are stations left
            if available_stations.shape[0] == 0:
                break

        # Split cluster data to train and test
        test_size = count * test_ratio
        test_stations = split_data(stations, test_size, vals_per_station)
        stations = np.array(stations)
        dataset = np.array(['train'] * len(stations))
        cond = np.isin(stations, test_stations)
        dataset[cond] = 'test'

        # Add the cluster stations to the dataframe list
        df_list.append(pd.DataFrame({
            'st_num': stations,
            'iter': iter_num,
            'cluster': cluster_number,
            'dataset': dataset}))

    # Concatenate dfs
    df = pd.concat(df_list, ignore_index=True)

    # Assigining remaining stations randomly
    st_left = available_stations.shape[0]
    if st_left != 0:
        stations = available_stations.values
        cluster_numbers = np.random.choice(df['cluster'].unique(), replace=True, size=st_left)
        dataset = np.random.choice(df['dataset'].unique(), replace=True, size=st_left)
        df = df.append(pd.DataFrame({
            'st_num': stations,
            'iter': iter_num,
            'cluster': cluster_numbers,
            'dataset': dataset}), ignore_index=True)

    return df


def permutate_clusters(n, all_data):
    '''
    Create n permutations of the regions in all_data.
    Steps:
        1. Create an array of cluster sizes using the function get_cluster_sizes
        2. Iterate through the cluster sizes array and assign stations to each cluster
           using the function create_clusters

    '''

    # Dirichlet distribution parameter
    beta = 95

    # Number of clusters
    n_clusters = all_data['region'].unique().shape[0]

    # Computing the number of data per cluster
    original_cluster_sizes = all_data['region'].value_counts().values
    n_data = all_data.shape[0]

    # Each row of this array will be filled by n_cluster values
    # where each value represents the number of data points in that cluster
    cluster_dist_arr = np.zeros((n, n_clusters))

    for i in range(n):
        cluster_dist_arr[i] = get_cluster_sizes(beta, original_cluster_sizes, n_data)

    # Number of datapoints per station
    vals_per_station = all_data['st_num'].value_counts()

    df_list = []

    # Assigning stations to clusters
    for i in range(n):
        vals_per_cluster = cluster_dist_arr[i]
        df_list.append(create_clusters(vals_per_station, vals_per_cluster, iter_num=i + 1))

    df = pd.concat(df_list, ignore_index=True)

    return df


# RandomizationTest class

class RandomizationTest:
    '''

    Parameters:
    -----------

    data: pandas.DataFrame
        Dataframe containing all variables from all stations

    permutations: pandas.DataFrame
        Dataframe containing all permutations of stations. (See creating_test_permutations.ipynb)
        Should include the following columns:
            - st_num
            - iter: Number of the permutation / iteration
            - cluster
            - dataset: train or test

    input_combo: int
        Number of the input combination as defined in the input combinations configuration file

    regressor: Regressor class
        Regressor class to use for fitting models in each permutation.
        Should have fit() and predict() methods

    score_fun: function
        Function to use for scoring models.
        Should take two arguments: y_true, y_pred
    '''

    def __init__(self,
                 data,
                 permutations,
                 input_combo,
                 regressor,
                 score_fun):

        self.data = data.copy()
        self.permutations = permutations.copy()
        self.input_combo = input_combo
        self.regressor = regressor
        self.score_fun = score_fun

        # Loading variable name/abbreviation dictionary
        with open('config_files/variable_abbreviations.yaml', 'r') as f:
            var_abbr_dict = yaml.safe_load(f)['VAR_NAMES']
        var_name_dict = {val: key for key, val in var_abbr_dict.items()}
        self.var_name_dict = var_name_dict
        self.var_abbr_dict = var_abbr_dict

        # Initializing results dataframe
        self.n_iter = permutations['iter'].max()
        self.n_cluster = permutations['cluster'].unique().shape[0]

        index = pd.MultiIndex.from_product([range(1, self.n_iter + 1), range(self.n_cluster + 1)],
                                           names=['iter', 'cluster'])
        self.result_df = pd.DataFrame(columns=['general_model', 'cluster_model'], index=index)

    def get_features(self,
                     input_combo,
                     path_to_input_combos_file='config_files/input_combinations.yaml'):
        """
        Get the input combinations defined in the input_combinations.yaml file.
        """

        # Getting features from the feature combos dictionary
        assert os.path.isfile(path_to_input_combos_file), \
            'input_combinations YAML file not found at {}'.format(path_to_input_combos_file)

        with open(path_to_input_combos_file, 'r') as file:
            input_combos = yaml.safe_load(file)['INPUT_COMBOS']

        assert input_combo in input_combos.keys(), \
            'Input combination {} is not defined in the input_combinations YAML file'.format(input_combo)

        feature_names = input_combos[input_combo]

        features = []
        for feature_name in feature_names:
            features.append(self.var_abbr_dict.get(feature_name, feature_name))

        return features

    def get_train_test(self, iter_num, cluster):
        '''
        Returns the training and test set of a certain cluster in a certain permutation.
        If cluster is 0, the training and test sets of the combined model are returned.

        '''

        # Slicing the defined iteration number
        cond = self.permutations['iter'] == iter_num

        if cluster != 0:
            # Slicing the defined cluster if not None
            cond = cond & (self.permutations['cluster'] == cluster)

        # Getting train and test stations
        train_cond = self.permutations['dataset'] == 'train'
        train_stations = self.permutations.loc[cond & train_cond, 'st_num'].to_numpy()
        test_stations = self.permutations.loc[cond & (~train_cond), 'st_num'].to_numpy()

        # Slicing train and test data from the defined dataset
        cond = self.data['st_num'].isin(train_stations)
        train = self.data[cond]

        cond = self.data['st_num'].isin(test_stations)
        test = self.data[cond]

        self.features = self.get_features(self.input_combo)
        cols = ['st_num'] + self.features + ['ET0']

        return (train[cols], test[cols])

    def get_Xy(self, iter_num, cluster):
        """
        Returns (X_train, X_test, y_train, y_test)
        """

        # Loading train and test datasets
        train, test = self.get_train_test(iter_num, cluster)

        X_train = train[self.features].to_numpy()
        X_test = test[self.features].to_numpy()

        # Checking if X has 1 dimension only
        if len(self.features) < 2:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        y_train = train['ET0'].to_numpy()
        y_test = test['ET0'].to_numpy()

        return (X_train, X_test, y_train, y_test)

    def run_permutation(self, iter_num):

        '''
        Run one permutation (defined by iter_num).
        '''

        cluster = 0

        # Loading train and test datasets
        train, test = self.get_train_test(iter_num, cluster)
        (X_train, X_test, y_train, y_test) = self.get_Xy(iter_num, cluster)

        predictions_df = test[['st_num', 'ET0']].copy()
        predictions_df.insert(1, 'cluster', -1)

        # Training model and getting predictions
        model = self.regressor()
        model.fit(X_train, y_train)

        # Saving predictions
        y_hat = model.predict(X_test)
        predictions_df.loc[:, 'general_pred'] = y_hat
        predictions_df.loc[:, 'cluster_pred'] = -100.

        # Saving general_model score
        self.result_df.loc[(iter_num, cluster), 'general_model'] = self.score_fun(y_test, y_hat)

        for cluster in np.arange(1, self.n_cluster + 1):
            # Loading train and test datasets
            train, test = self.get_train_test(iter_num, cluster)
            (X_train, X_test, y_train, y_test) = self.get_Xy(iter_num, cluster)

            # Training model and getting predictions
            model = self.regressor()
            model.fit(X_train, y_train)

            # Saving predictions
            y_hat = model.predict(X_test)
            idx = test.index
            predictions_df.loc[idx, 'cluster_pred'] = y_hat
            predictions_df.loc[idx, 'cluster'] = cluster

            # Saving score of the cluster model
            self.result_df.loc[(iter_num, cluster), 'cluster_model'] = self.score_fun(y_test, y_hat)

            # Saving general_model score for this cluster
            y_hat = predictions_df.loc[idx, 'general_pred']
            self.result_df.loc[(iter_num, cluster), 'general_model'] = self.score_fun(y_test, y_hat)

        # Saving total cluster_model score
        y_test = predictions_df['ET0']
        y_hat = predictions_df['cluster_pred']
        self.result_df.loc[(iter_num, 0), 'cluster_model'] = self.score_fun(y_test, y_hat)

    def run_test(self):

        print('Started randomization test')
        print(f'Number of iterations: {self.n_iter}')
        print('\n')
        print('_' * 49)

        # Loading steps
        loading_steps = np.linspace(0, self.n_iter + 1, 50).astype('int')
        loading_steps = iter(loading_steps)
        next_step = next(loading_steps)

        # Number of iterations before taking a break
        iter_between_breaks = 50

        # Length of break (sec)
        break_len = 0

        for iter_num in range(1, self.n_iter + 1):
            self.run_permutation(iter_num)

            if iter_num % iter_between_breaks == 0:
                time.sleep(break_len)

            if iter_num > next_step:
                print('#', sep='', end='')
                next_step = next(loading_steps)

        return self.result_df