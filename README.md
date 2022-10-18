# General and regional cross‑station assessment of machine learning models for estimating reference evapotranspiration

This is the code of the paper "General and regional cross‑station assessment of machine learning models for estimating reference evapotranspiration", which is published in Acta Geophysica (DOI: [10.1007/s11600-022-00939-9](https://link.springer.com/article/10.1007/s11600-022-00939-9)).


## Abstract
Significant research has been done on estimating reference evapotranspiration (ET0) from limited climatic measurements using machine learning (ML) to facilitate the acquirement of ET0 values in areas with limited access to weather stations. However, the spatial generalizability of ET0 estimating ML models is still questionable, especially in regions with significant climatic variation like Turkey. Aiming at exploring this generalizability, this study compares two ET0 modeling approaches: (1) one general model covering all of Turkey, (2) seven regional models, one model for each of Turkey’s seven regions. In both approaches, ET0 was predicted using 16 input combinations and 3 ML methods: support vector regression (SVR), Gaussian process regression (GPR), and random forest (RF). A cross-station evaluation was used to evaluate the models. Results showed that the use of regional models created using SVR and GPR methods resulted in a reduction in root mean squared error (RMSE) in comparison with the general model approach. Models created using the RF method suffered from overfitting in the regional models’ approach. Furthermore, a randomization test showed that the reduction in RMSE when using these regional models was statistically significant. These results emphasize the importance of defining the spatial extent of ET0 estimating models to maintain their generalizability.


## Tableau dashboards
Interactive Tableau dashboards visualizing the input data and modeling results were created to allow readers to explore the data easily:
* [Input data visualization](https://public.tableau.com/views/ETDashboard/Story1?:language=en-US&:display_count=n&:origin=viz_share_link)
* [Modeling results visualization](https://public.tableau.com/views/ETResultsEvaluation/ResultAnalysis?:language=en-US&:display_count=n&:origin=viz_share_link)


## Repository description
The code is divided into directories, each corresponding to one process in the project as described below. Further description of the directories and their contents can be found in the README files within each directory. The classes and functions used in this study are saved in **ETProject** and loaded from there when applicable.

Directory description (ordered by the worklflow of the study):
* [config_files](https://github.com/YZouzou/et0-estimation/tree/main/config_files#configuration-files): Contains all the configuration files used in the code.
* [ETProject](https://github.com/YZouzou/et0-estimation/tree/main/ETProject#etproject): A package that includes all functions and classes used in this project.
* [data_preparation](https://github.com/YZouzou/et0-estimation/tree/main/data_preparation#data-preparation): Includes the code used for cleaning and preparing the raw weather data received from the meteorological services.
* [EDA](https://github.com/YZouzou/et0-estimation/tree/main/EDA#exploratory-data-analysis): Contains an exploratory data analysis of the data used in this study.
* [models](https://github.com/YZouzou/et0-estimation/tree/main/models#models): Includes the trained models, their parameters, their predictions, and model performance metrics. This directory also includes a notebook exploring model performances extensively.
* [randomization_test](https://github.com/YZouzou/et0-estimation/tree/main/randomization_test#randomization-test): Includes the code used to conduct the randomization test.


## Requirements
1. Clone this repository.

2. Run `pip install -r requirements.txt` (preferably in a virtual environment).

This will install all the required packages except GeoPandas. Installing GeoPandas is a cumbersome process (see installation steps [here](https://geopandas.org/en/stable/getting_started/install.html) and [here](https://geoffboeing.com/2014/09/using-geopandas-windows/)). However, in this code GeoPandas is only used for visualization and it is not necessary to run the other parts of the code (cleaning data, training and running models, etc.).

## Data availability
The data obtained from the Turkish meteorological services can only be shared on reasonable request. The data can be explored in the interactive Tableau dashboards mentioned above.
