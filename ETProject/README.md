# ETProject
This package includes all functions and classes used in this project.

## ET0_functions
This module contains the definitions of all the functions required for ET0 computation.

## DataProcessor
This module contains the definition of the `DataProcessor` class, which is used in **process_data.py** in *data_preparation/* to process the cleaned data and create the **et_data.csv** file.

## EDA
This module contains the functions used in exploratory data analysis (EDA)

## MyModel
This module contains the class `MyModel` along with the classes of each regressor (SVR, GPR, RF, and Polynomial).

### `MyModel` class
This class is used to train models and save their scores, predictions, and parameters in a systematic way to be easily accessible for further model result exploration.

### How this class works:

#### Case 1: Creating a new model:

1. A regressor class should be created that has the properties:
    * `regressor` (The regressor class) is initialized using the parameters `model_params` (dict), which is an input parameter of the `MyModel` class.
    * Has the following methods:
        * `regressor.fit(X_train, y_train, **fit_params)`. Where `fit_params` (dict) is an input parameter of the `MyModel` class. This method should fit the regressor to the defined training data (`X_train` and `y_train`).
        * `regressor.predict(X_test)`. This method returns the predictions of the input `X_test`.
        * `regressor.get_params()`. Returns a dictionary of the regressor parameter names and values.
    * Has the following attributes:
        * `train_time` which stores the training time of the model.
        * `algorithm` which stores the name of the algorithm.
    
2. Define the path to the directory where the models, predictions, evaluation metrics, and model parameters will be saved (`save_path`). This directory should include four sub-directories:
    1. *model_results/* which will include model scores and station scores.
    2. *model_parameters/* which will include the parameters of each model, one file for each algorithm used.
    3. *model_predictions/* which will include the predictions of each model named using the model name and saved as a .npz file.
    4. *trained_models* which will include the trained models.
    
3. Define the path to the directory containing the processed dataset (`data_path`).

4. Define a function dictionary (`score_fun_dict`) containing the metric names and functions should be defined (metric_name -> metric_fun), where the metric function takes two arguments: `y_true` and `y_pred`.

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

8. Run the `save()` method to save the model, model parameters, model evaluation scores, predictions, and station scores. 
    
#### Case 2: Loading a trained model
1. Define `region`, `input_combo`, and `algorithm`
2. Initialize a MyModel object using the three aforementioned arguments, the model will be automatically loaded if found in the relative directory.

## test_MyModel
This script is for testing the class `MyModel`.

## ModelEvaluation
This module contains the functions used for exploring the performances of models.

## RandomizationTest
This module contains the functions used to create permuations for the randomization test, along with the `RandomizationTest` class used to conduct the randomization test.