from MyModel import MyModel, MLRRegressor, SVRRegressorRBF, PolynomialRegressor, RFRegressor
import os
from pathlib import Path
import numpy as np
import pandas as pd

# MyModel class is tested in this script
# If True the test results are not deleted and can be found in __MyModelTest__/
keep_files = False

# Algorithms to be tested
algorithm_dict = {'MLR': MLRRegressor, 'SVR': SVRRegressorRBF, 'Polynomial': PolynomialRegressor, 'RF': RFRegressor}
fit_params_dict = {'MLR': {}, 'SVR': {'C': [1, 5], 'gamma': [0.5, 1], 'cv': 5}, 'Polynomial': {},
                   'RF': {'n_estimators': [50, 100], 'min_samples_leaf': [50, 100], 'max_samples': [0.5, 0.75]}}
model_params_dict = {'MLR': {}, 'SVR': {}, 'Polynomial': {}, 'RF': {}}

# Loading data
main_dir_path = Path.cwd().parents[0]
os.chdir(main_dir_path)
et_df = pd.read_csv('processed_data/et_data.csv')

# Creating directory to save the model results
dir_list = ['__MyModelTest__/model_results', '__MyModelTest__/model_parameters',
            '__MyModelTest__/model_predictions', '__MyModelTest__/trained_models']

if not os.path.exists('__MyModelTest__'):
    os.mkdir('__MyModelTest__')
for directory in dir_list:
    if os.path.exists(directory):
        for file in Path(directory).glob('*'):
            os.remove(file)
    else:
        os.mkdir(directory)

# Creating mock dataset
df_list = []
for region in et_df['region'].unique():
    cond1 = et_df['region'] == region
    cond2 = et_df['dataset'] == 'train'
    # Training data
    cond = cond1 & cond2
    df_list.append(et_df.loc[cond].sample(n=200))
    # Test data
    cond = cond1 & (~cond2)
    df_list.append(et_df.loc[cond].sample(n=100))

df = pd.concat(df_list, ignore_index=True)

# Dropping stations with one occurrence only to avoid warning when running model.get_station_scores()
cond = df['st_num'].value_counts() == 1
st_to_drop = cond[cond].index
cond = df['st_num'].isin(st_to_drop)
df = df.drop(index=df[cond].index)
df.to_csv('__MyModelTest__/mock_data.csv', index=False)


def test1(region_name, input_combination, regressor, model_params, fit_params):
    """
    Tests the MyModel by creating a model for a random region and random input combination.

    """

    model = MyModel(
        region=region_name,
        input_combo=input_combination,
        regressor=regressor,
        model_params=model_params,
        fit_params=fit_params,
        save_path='__MyModelTest__/',
        data_path='__MyModelTest__/mock_data.csv'
    )

    model.fit_model()
    model.save()


def test2(region_name, input_combination, algorithm):
    """
    Tests the MyModel class by loading a previously created model

    """
    model = MyModel(
        region=region_name,
        input_combo=input_combination,
        algorithm=algorithm,
        save_path='__MyModelTest__/',
        data_path='__MyModelTest__/mock_data.csv'
    )

    # Checking attributes
    for attr in ['features', 'feature_names', 'X_train', 'X_test', 'y_train', 'y_test',
                 'train_stations', 'test_stations', 'y_hat_train', 'y_hat_test']:
        assert hasattr(model, attr), f"{attr} not assigned to MyModel object"

    # Checking methods
    model.get_scores(print_vals=False, predict=False)
    model.get_scores(print_vals=False, predict=True)
    _ = model.get_station_scores()

    # Comparing loaded model predictions with saved predictions
    y_pred = model.model.predict(model.X_train)
    assert all(np.isclose(y_pred, model.y_hat_train)), "Train predictions different from saved predictions"
    y_pred = model.model.predict(model.X_test)
    assert all(np.isclose(y_pred, model.y_hat_test)), "Train predictions different from saved predictions"


# Running tests
created_models = {}
try:
    for algorithm, regressor in algorithm_dict.items():
        region_name = np.random.choice(et_df['region'].unique())
        input_combination = np.random.randint(1, 17)
        model_params = model_params_dict[algorithm]
        fit_params = fit_params_dict[algorithm]

        # Added the created model to a dictionary to load it in the next test
        created_models[algorithm] = (region_name, input_combination)

        test1(region_name, input_combination, regressor=regressor, model_params=model_params, fit_params=fit_params)
    print('\nTest 1: Successful')

except Exception as e:
    print('\nTest 1: Failed')
    print(f'Reason: {e}')

try:
    for algorithm, (region_name, input_combination) in created_models.items():
        test2(region_name, input_combination, algorithm)
    print('\nTest 2: Successful')

except Exception as e:
    print('\nTest 2: Failed')
    print(f'Reason: {e}')

if not keep_files:
    for directory in dir_list:
        for file in Path(directory).glob('*'):
            os.remove(file)
        os.rmdir(directory)

    os.remove('__MyModelTest__/mock_data.csv')
    os.rmdir('__MyModelTest__')
