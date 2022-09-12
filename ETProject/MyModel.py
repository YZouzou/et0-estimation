import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os
import yaml
from joblib import dump, load
import time
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter

# SVGPR functions
def run_adam(model,
             train_dataset,
             minibatch_size,
             iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    # Loading steps
    loading_steps = np.linspace(0, iterations, 40).astype('int')
    loading_steps = iter(loading_steps)
    next_step = next(loading_steps)

    print('Training model..')
    print('_' * 40)
    for step in range(iterations):

        if step >= next_step:
            print('#', sep='', end='')
            next_step = next(loading_steps)

        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)

    print('\n')
    return logf


class SVGPRegressorRBF:
    '''
    Wrapping class for the GPFlow SVGPR model.
    This class is for a SVGPR model using a RBF kernel and a Gaussian likelihood.
    '''

    def __init__(self):
        self.algorithm = 'GPR'

    def fit(self,
            X_train,
            y_train,
            M=0.1,
            minibatch_size=100,
            iterations=10000,
            delay=0):
        '''
        Fit the model to the given data.

        Parameters:
        -----------

        X_train, y_train: np.ndarray
            Training data to train model on.

        M: int or float between 0,1
            Number of inducing variables.
            The inducing variables are randomly chosen from the data.
            If M is a float between 0 and 1, a proportion equal to M
            from the training data is used as inducing variables.

        minibatch_size: int

        iterations: int

        delay: int
            Number of seconds to delay the fitting.
            When fitting various clusters using the MasterModel, it may be useful to add
            a delay to avoid continuous CPU utilization.
        '''

        # Reshaping target vector
        y_train = y_train.reshape(-1, 1)

        # Scaling data
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.scaler = scaler

        # Transforming training data
        X_train = scaler.transform(X_train)
        data = (X_train, y_train)

        n, input_dim = X_train.shape

        tensor_data = tuple(map(tf.convert_to_tensor, data))

        train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(n)

        # Kernel
        lengthscales = np.ones(input_dim)
        k = gpflow.kernels.RBF(lengthscales=lengthscales)

        # Checking if M is a ratio
        if M <= 1:
            M = int(M * n)

        # Initialize inducing locations
        rand_idx = np.random.randint(0, n, size=M)
        Z = X_train[rand_idx, :]

        m = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(), Z, num_data=n)

        gpflow.set_trainable(m.inducing_variable, False)

        t1 = time.time()

        logf = run_adam(model=m, train_dataset=train_dataset,
                        minibatch_size=minibatch_size, iterations=iterations)

        t2 = time.time()

        train_time = t2 - t1

        self.model = m
        self.train_time = train_time
        self.M = M
        self.batch_size = minibatch_size

        print('Training done, training time: {:.1f}'.format(train_time))
        print('{} second delay'.format(delay))
        time.sleep(delay)

    #         return logf

    def predict(self, X_test):
        X = self.scaler.transform(X_test)
        mean, var = self.model.predict_f(X)

        return mean.numpy().flatten()  # , var

    def get_params(self):
        '''
        Returns a dictionary of the model parameters

        SVGPR RBF parameters:
            kernel_var: kernel variance
            kernel_len: kernel lengthscales
            likelihood_var: likelihood variance
            M: Number of inducing variables
            batch_size: Batch size used in stochastic gradient descent
        '''
        m = self.model

        param_dict = {}
        # Kernel variance
        param_dict['kernel_var'] = m.kernel.variance.numpy() * 1

        # Lengthscales
        lengthscales = m.kernel.lengthscales.numpy()
        # Converting list to a comma seperated string
        # to simplify dataframe unpacking
        # (A list is saved as a string in a CSV file)
        lengthscales = list(map(str, lengthscales))
        lengthscales = ','.join(lengthscales)
        param_dict['kernel_len'] = lengthscales

        # Likelihood variance
        param_dict['likelihood_var'] = m.likelihood.variance.numpy() * 1

        # Num. inducing variables
        param_dict['M'] = self.M

        # Batch size
        param_dict['batch_size'] = self.batch_size

        return param_dict


class PolynomialRegressor:

    def __init__(self):
        self.algorithm = 'Polynomial'

    def fit(self, X_train, y_train):
        """
        Fit the model to the given data
        """

        steps = [
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('lr', LinearRegression())
        ]

        pipe = Pipeline(steps)

        t1 = time.time()
        pipe.fit(X_train, y_train)
        t2 = time.time()
        train_time = t2 - t1

        self.train_time = train_time
        self.model = pipe

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)

        return y_hat

    def get_params(self):
        """
        Returns a dictionary of the model parameters
        """

        return {}


class RFRegressor:

    def __init__(self):
        self.algorithm = 'RF'

    def create_grid(self, n_estimators, min_samples_leaf, max_samples, cv):
        """
        Create a grid search for a SVR model with RBF kernel
        """

        pipe = RandomForestRegressor(n_jobs=2)

        param_grid = {'n_estimators': n_estimators,
                      'min_samples_leaf': min_samples_leaf,
                      'max_samples': max_samples
                      }

        scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}

        grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                            scoring=scoring, refit='R2', cv=cv, verbose=2)

        return pipe, grid

    def fit(self,
            X_train,
            y_train,
            n_estimators=[50, 100, 300],
            min_samples_leaf=[10, 20, 50, 100],
            max_samples=[0.5, 0.75, 1],
            cv=5):
        """
        Find the best parameters using grid search and fit the model using these parameters.
        """
        # Creating pipeline and grid search
        pipe, grid = self.create_grid(n_estimators, min_samples_leaf, max_samples, cv)

        # Fitting grid search
        grid.fit(X_train, y_train)

        # Updating pipeline parameters
        pipe.set_params(**grid.best_params_)

        t1 = time.time()
        pipe.fit(X_train, y_train)
        t2 = time.time()
        train_time = t2 - t1

        self.train_time = train_time
        self.model = pipe

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)

        return y_hat

    def get_params(self):
        """
        Returns a dictionary of the model parameters
        """

        params = ['n_estimators', 'min_samples_leaf', 'max_samples']
        param_dict = {}

        for param in params:
            param_dict[param] = getattr(self.model, param)

        return param_dict


class SVRRegressorRBF:

    def __init__(self):
        self.algorithm = 'SVR'

    def create_svr_grid(self, C, gamma, cv):
        """
        Create a grid search for a SVR model with RBF kernel
        """

        steps = [
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', cache_size=500))
        ]

        pipe = Pipeline(steps=steps)

        param_grid = {'svr__C': C,
                      'svr__gamma': gamma}

        scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
        grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                            scoring=scoring, return_train_score=True,
                            refit='R2', cv=cv, verbose=2)
        return pipe, grid

    def fit(self, X_train, y_train, C=[1, 5, 50, 100], gamma=[0.05, 0.1, 0.5, 1], cv=5):
        """
        Find the best parameters using grid search and fit the model using these parameters.
        """
        # Creating pipeline and grid search
        pipe, grid = self.create_svr_grid(C, gamma, cv)

        # Fitting grid search
        grid.fit(X_train, y_train)

        # Updating pipeline parameters
        pipe.set_params(**grid.best_params_)

        t1 = time.time()
        pipe.fit(X_train, y_train)
        t2 = time.time()
        train_time = t2 - t1

        self.train_time = train_time
        self.model = pipe

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)

        return y_hat

    def get_params(self):
        """
        Returns a dictionary of the model parameters
        SVR RBF parameters: kernel, C, gamma
        """

        params = ['kernel', 'C', 'gamma']
        param_dict = {}

        for param in params:
            param_dict[param] = getattr(self.model['svr'], param)

        return param_dict


class MLRRegressor:

    def __init__(self):
        self.algorithm = 'MLR'

    def fit(self, X_train, y_train):
        """
        Fit the MLR model to the given data
        """

        steps = [
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]

        pipe = Pipeline(steps)

        t1 = time.time()
        pipe.fit(X_train, y_train)
        t2 = time.time()
        train_time = t2 - t1

        self.train_time = train_time
        self.model = pipe

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)

        return y_hat

    def get_params(self):
        """
        Returns a dictionary of the model parameters
        """

        return {}


class MyModel:
    """
This class is used to train models and save their scores, predictions, and parameters in a systematic way to be
easily accessible for further model result exploration.


**Case 1: Creating a new model:**

    1. A regressor class should be created that has the properties:
        1. `regressor` (The regressor class) is initialized using the parameters `model_params` (dict), which is an \
        input parameter of the `MyModel` class.

        2. Has the following methods:
            - `regressor.fit(X_train, y_train, **fit_params)`. Where `fit_params` (dict) is an input parameter of the
              `MyModel` class. This method should fit the regressor to the defined training data
              (`X_train` and `y_train`).
            - `regressor.predict(X_test)`. This method returns the predictions of the input `X_test`.
            - `regressor.get_params()`. Returns a dictionary of the regressor parameter names and values.

        3. Has the following attributes:
            * `train_time` which stores the training time of the model.
            * `algorithm` which stores the name of the algorithm.

    2. Define the path to the directory where the models, predictions, evaluation metrics, and model parameters will be saved (`save_path`). This directory should include four sub-directories:
        * *model_results/* which will include model scores and station scores.
        * *model_parameters/* which will include the parameters of each model, one file for each algorithm used.
        * *model_predictions/* which will include the predictions of each model named using the model name and
          saved as a .npz file.
        * *trained_models* which will include the trained models.

    3. Define the path to the directory containing the processed dataset (`data_path`).

    4. Define a function dictionary (`score_fun_dict`) containing the metric names and functions should be defined
       (metric_name -> metric_fun), where the metric function takes two arguments: `y_true` and `y_pred`.

    5. Define the input combinations in the yaml file stored at *config_files/input_combinations.yaml*.

    6. After defining the aforementioned items, a `MyModel` object is initialized by defining the following arguments:
        * **region**: *str* Region name. If None, all data is used.
        * **input_combo**: *int* Number of the input combination defined in the input_combinations.yaml file.
        * **regressor**: *regressor object*
        * **model_params**: *dict* Regressor object init parameters
        * **fit_params**: *dict* Regressor fit parameters
        * **save_path**: *str*
        * **data_path**: *str*
        * **target**: *str* Target variable. Default is 'ET0'
        * **score_fun_dict** *dict*

    7. Run the `fit_model()` method to initialize and fit the defined regressor model.

    8. Run the `save()` method to save the model, model parameters, model evaluation scores, predictions,
       and station scores.


**Case 2: Loading a trained model:**
    1. Define `region`, `input_combo`, and `algorithm`.
    2. Initialize a MyModel object using the three aforementioned arguments, the model will be automatically loaded if found in the relative directory.

Parameters:
-----------

    region: str
        Region name. If None, all data is used.

    input_combo: int
        Number of the feature combination defined in the input_combinations.yaml file

    algorithm: str
        Algorithm name. Only required if loading a saved model.

    regressor:
        Regressor object

    model_params: dict
        Regressor object init parameters

    fit_params: dict
        Regressor fit parameters

    target:
        Target variable. Default is 'ET0'

    save_path:
        Path to the directory where the model and predictions are saved.

    data_path:
        Path to the directory where the data is saved

    score_fun_dict: dict
        A dictionary of metric functions to be used for evaluating the model


Methods:
--------

    m.get_features():
        Get the full feature names and abbreviations.
        Used internally to create the attributes m.features and m.feature_names

    m.get_train_test():
        Get training dataset from the defined data_path.
        Returns train, test

    m.get_Xy():
        Returns X_train, X_test, y_train, y_test

    m.get_scores():
        Compute evaluation metrics.
        Predictions and metrics are assigned to attributes

    m.get_station_scores():
        Get scores per weather station.

    m.save():
        Save model, model predictions, model scores, model parameters, and station scores
        to the defined save path.
    """

    def __init__(self,
                 region=None,
                 input_combo=None,
                 algorithm=None,
                 regressor=None,
                 model_params=None,
                 fit_params=None,
                 save_path='models/',
                 data_path='processed_data/et_data.csv',
                 target='ET0',
                 score_fun_dict={
                     'MAE': mean_absolute_error,
                     'RMSE': lambda x, y: np.sqrt(mean_squared_error(x, y)),
                     'rRMSE': lambda x, y: np.sqrt(mean_squared_error(x, y)) / np.mean(x),
                     'NSE': r2_score}):

        if region is None:
            region = 'all_data'

        self.region = region

        self.data_path = data_path

        if regressor is not None:
            self.regressor = regressor
            algorithm = regressor().algorithm
        else:
            assert algorithm is not None,\
                "Define a regressor to create a new model or enter the algorithm name to load an existing model"

        # Creating model name
        self.model_name = f"{algorithm}_{region.replace(' ', '_').lower()}_f{input_combo}"

        self.save_path = save_path
        self.model_path = os.path.join(save_path, 'trained_models', self.model_name + '.joblib')
        self.pred_path = os.path.join(save_path, 'model_predictions', self.model_name + '.npz')

        self.score_fun_dict = score_fun_dict

        self.input_combo = input_combo

        self.model_params = model_params

        self.fit_params = fit_params

        # Target variable
        self.target = target

        # Loading variable name/abbreviation dictionary
        with open('config_files/variable_abbreviations.yaml', 'r') as f:
            var_abbr_dict = yaml.safe_load(f)['VAR_NAMES']
        var_name_dict = {val: key for key, val in var_abbr_dict.items()}
        self.var_name_dict = var_name_dict
        self.var_abbr_dict = var_abbr_dict

        # Load model if saved and print a message if not found
        if self.load_model_():
            print('{} model loaded from {}'.format(self.model_name, save_path))
            return

        # Else if the model is not saved
        self.features, self.feature_names = self.get_features(input_combo)

        # Assigning train and test data attributes
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_Xy()

        # Assigning station numbers and dates
        train, test = self.get_train_test()
        self.train_stations = train[['st_num', 'date']]
        self.test_stations = test[['st_num', 'date']]

        print('Model not found at {}'.format(self.model_path))
        print('MyModel object created without model')

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

        return features, feature_names

    def get_train_test(self, how='selected_features'):
        """
        Get training dataset from the defined data_path.
        Returns the tuple (train, test)

        Parameters:
        -----------

        how:
            'all' return all training dataset with all features
            'selected_features' return training dataset with selected features only

        """
        data = pd.read_csv(self.data_path)

        # Slicing the defined region
        if self.region != 'all_data':
            cond = (data['region'] == self.region)
            data = data.loc[cond]

        # Adding date column
        data['day'] = 15
        data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
        data = data.drop(columns=['day'])

        cond = data['dataset'] == 'train'
        train = data.loc[cond]
        test = data.loc[~cond].reset_index(drop=True)

        # Shuffling training data
        train = train.sample(frac=1, random_state=12).reset_index(drop=True)

        if how == 'all':
            return train, test

        elif how == 'selected_features':
            cols = ['st_num', 'date'] + self.features + [self.target]
            return train[cols], test[cols]

    def get_Xy(self):
        """
        Returns (X_train, X_test, y_train, y_test)
        """

        # Loading train and test datasets
        train, test = self.get_train_test(how='selected_features')

        X_train = train[self.features].to_numpy()
        X_test = test[self.features].to_numpy()

        # Checking if X has 1 dimension only
        if len(self.features) < 2:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        y_train = train[self.target].to_numpy()
        y_test = test[self.target].to_numpy()

        return X_train, X_test, y_train, y_test

    def add_model_(self, model, train_time):
        """
        Assign model to attribute m.model
        """
        self.model = model
        self.train_time = train_time

    def fit_model(self, print_vals=True):
        """
        Initialize the regressor and fit it to training data using the defined model_params and fit_params

        Parameters:
        ------------

        print_vals: Bool
            Whether to print the model scores or not

        """

        model = self.regressor(**self.model_params)

        model.fit(self.X_train, self.y_train, **self.fit_params)

        self.add_model_(model, model.train_time)

        self.get_scores(predict=True, print_vals=print_vals)

    def load_model_(self):
        """
        Load model and predictions from the defined model_path and assign it to attribute m.model
        and returns True.
        If the model is not available at the defined path, returns False.
        """
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)
            preds = np.load(self.pred_path)

            for pred in ['y_hat_train', 'y_hat_test']:
                setattr(self, pred, preds[pred])

            # Getting the list of features used
            self.features, self.feature_names = self.get_features(self.input_combo)

            # Assigning train and test data attributes
            self.X_train, self.X_test, self.y_train, self.y_test = self.get_Xy()

            # Assigning station numbers and dates
            train, test = self.get_train_test()
            self.train_stations = train[['st_num', 'date']]
            self.test_stations = test[['st_num', 'date']]

            return True
        else:
            return False

    def get_scores(self,
                   print_vals=True,
                   predict=False):

        """
        Predict y_train and y_test then compute evaluation metrics.
        Predictions and metrics are assigned to attributes:

        self.y_hat_train: Training set predictions
        self.y_hat_test: Test set predictions

        self.train_scores: Dictionary of training eval. metrics
        self.test_scores: Dictionary of test eval. metrics

        Parameters:
        -----------

        print_vals: Print evaluation metrics if True

        predict: Predict from data if True
                 Use saved predictions if False
        """

        # Getting train and test predictions
        if predict is True:
            self.y_hat_test = self.model.predict(self.X_test)
            self.y_hat_train = self.model.predict(self.X_train)
        else:
            assert (hasattr(self, 'y_hat_train') and hasattr(self, 'y_hat_test')), \
                'Load predictions or set predict = True to predict target variable'

        train_scores = {}
        test_scores = {}

        for metric, fun in self.score_fun_dict.items():
            train_scores[metric] = fun(self.y_train, self.y_hat_train)
            test_scores[metric] = fun(self.y_test, self.y_hat_test)

        # Printing scores
        if print_vals is True:
            print('Train scores:')
            for metric, val in train_scores.items():
                print('{} = {:.3f}'.format(metric, val))

            print('\n')
            print('Test scores:')
            for metric, val in test_scores.items():
                print('{} = {:.3f}'.format(metric, val))

        self.train_scores = train_scores
        self.test_scores = test_scores

    def get_station_scores(self):
        """
        Get scores for every station to compare the performance of the model in different stations.
        """

        all_stations = np.concatenate([self.test_stations['st_num'].unique(),
                                       self.train_stations['st_num'].unique()])

        df = {
            'st_num': all_stations,
            'dataset': []
        }

        # Creating a dictionary of score_metric: empty list
        scores = {key: [] for key in self.score_fun_dict.keys()}
        df.update(scores)

        for st in all_stations:
            if st in self.test_stations['st_num'].unique():
                cond = self.test_stations['st_num'] == st
                # Extracting index of the station "st"
                idx = self.test_stations[cond].index
                y = self.y_test[idx]
                y_hat = self.y_hat_test[idx]
                df['dataset'].append('test')
            else:
                cond = self.train_stations['st_num'] == st
                # Extracting index of the station "st"
                idx = self.train_stations[cond].index
                y = self.y_train[idx]
                y_hat = self.y_hat_train[idx]
                df['dataset'].append('train')

            # Computing score metric values
            for metric, fun in self.score_fun_dict.items():
                df[metric].append(fun(y, y_hat))

        df = pd.DataFrame(df)

        # Adding algorithm name and model name

        df.insert(0, 'input_combo', self.input_combo)
        df.insert(0, 'region', self.region)
        df.insert(0, 'algorithm', self.model.algorithm)
        df.insert(0, 'model_name', self.model_name)

        return df

    def save_parameters_(self, model_parameters_dir_path='model_parameters'):
        """
        Save the parameters of the model to a CSV file.
        The method checks if the defined file_name is available in the defined directory,
        and appends the parameters to that file. Else, a new file is created.

        Parameters:
        -----------

        model_parameters_dir_path: str
            Path to the directory containing the saved model parameters.
        """

        file_path = os.path.join(self.save_path, model_parameters_dir_path, self.model.algorithm + '_parameters.csv')

        # Getting the parameter values from the model object
        param_dict = self.model.get_params()

        # Creating a dictionary of the new entries
        new_dict = {'model_name': self.model_name,
                    'algorithm': self.model.algorithm,
                    'region': self.region,
                    'input_combo': self.input_combo,
                    'train_time': self.model.train_time}

        new_dict.update(param_dict)

        # Checking if file is available
        if os.path.isfile(file_path) is not True:
            param_df = pd.DataFrame(new_dict, index=[0])
        else:
            param_df = pd.read_csv(file_path)
            param_df = param_df.append(pd.DataFrame(new_dict, index=[0]), ignore_index=True)
            param_df = param_df.drop_duplicates(subset=['model_name', 'algorithm', 'region', 'input_combo'],
                                                keep='last')

        param_df.to_csv(file_path, index=False)

    def save_scores_(self,
                     model_scores_file_path='model_results/model_scores.csv',
                     station_scores_file_path='model_results/station_scores.csv'
                     ):

        """
        Save model evaluation metrics to a CSV file. If the file exists values are appended to it
        and duplicates are dropped. Scores (evaluation metrics) per station are saved to the file defined in
        station_scores_file_name.

        Parameters:
        -----------

        scores_file_name: str
            Name of the file including total scores

        station_scores_file_name: str
            Name of the file including scores per station

        """

        train_dict = {
            'model_name': self.model_name,
            'algorithm': self.model.algorithm,
            'region': self.region,
            'input_combo': self.input_combo,
            'dataset': 'train'
        }

        train_dict.update(self.train_scores)

        test_dict = {
            'model_name': self.model_name,
            'algorithm': self.model.algorithm,
            'region': self.region,
            'input_combo': self.input_combo,
            'dataset': 'test'
        }

        test_dict.update(self.test_scores)

        model_scores_file_path = os.path.join(self.save_path, model_scores_file_path)

        # Checking if file is available
        if os.path.isfile(model_scores_file_path) is not True:
            score_df = pd.DataFrame([train_dict, test_dict])
        else:
            score_df = pd.read_csv(model_scores_file_path)
            score_df = score_df.append(pd.DataFrame([train_dict, test_dict]), ignore_index=True)
            score_df = score_df.drop_duplicates(subset=['model_name', 'algorithm', 'region',
                                                        'input_combo', 'dataset'], keep='last')

        score_df.to_csv(model_scores_file_path, index=False)

        # Saving station scores
        station_scores = self.get_station_scores()

        station_scores_file_path = os.path.join(self.save_path, station_scores_file_path)

        # Checking if file is available
        if not os.path.isfile(station_scores_file_path):
            station_scores.to_csv(station_scores_file_path, index=False)

        else:
            old_station_scores = pd.read_csv(station_scores_file_path)
            df = old_station_scores.append(station_scores, ignore_index=True)
            df = df.drop_duplicates(subset=['model_name', 'algorithm', 'region', 'st_num',
                                            'input_combo', 'dataset'], keep='last')
            df.to_csv(station_scores_file_path, index=False)

    def save_model_(self):
        """
        Save model and predictions to the defined model_path
        """
        # Saving model
        dump(self.model, self.model_path)

    def save_predictions_(self):
        """
        Saves model predictions.
        """
        assert(hasattr(self, 'y_hat_train') and hasattr(self, 'y_hat_test')), \
            'Predictions not available'

        preds = ['y_hat_train', 'y_hat_test']
        preds = dict([(p, getattr(self, p)) for p in preds])
        np.savez(self.pred_path, **preds)

    def save(self):
        """
        Save model, model predictions, model scores, model parameters, and station scores
        to the defined save path.
        """

        self.save_scores_()
        self.save_model_()
        self.save_parameters_()
        self.save_predictions_()
