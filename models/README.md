# Models
This directory includes the trained models, their parameters, their predictions, and model performance metrics. Models are named as follows: ***{algorithm}_{region}_f{input_combination}***. General models have region = "all_data".

The model predictions and their performance metrics are visualized in a Tableau dashboard [here](https://public.tableau.com/views/ETResultsEvaluation/ResultAnalysis?:language=en-US&:display_count=n&:origin=viz_share_link). (It is recommended to view the dashboard in full screen using the "Full Screen" button in the bottom right of the dashboard).

## Directory description:

### model_parameters
This directory contains the trained model parameters and the training times of each algorithm used in this project.

### model_predictions
This directory contains the model predictions of all algorithms, input combinations, and region/general models. The predictions are provided by the naming convention defined above.

### model_results
This directory contains the computed performance metrics of the trained models. A file for performance metrics by station is also provided.

### trained_models
This directory contains the trained models using the naming convention defined above.

### model_training.ipynb
In this notebook, the models are trained using the `MyModel` class.

### creating_region_general_score_dataset.ipynb
In this notebook, the predictions of all regional models are combined to get the predictions for all stations. Then, these predictions are used to compute the performance metrics of the regional models combined, which are comparable to the performance metrics computed for the general model. (Remember that the test stations used in regional and general scenarios are the same). Similarly, the general model predictions are separated to compute the performance metrics of the general model for each region, which are comparable to the regional models' performance metrics. The computed metrics are saved in **region_general_scores.csv**, where for each combination of `algorithm`, `region`, `input_combo`, and `dataset` a `region_metric` and a `general_metric` are provided for the regional models and the general model, respectively.

### performance_evaluation.ipynb
In this notebook, the model performances and predictions are visualized and compared.



