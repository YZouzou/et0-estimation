import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def load_predictions(algorithm, region, input_combo):
    """
    Returns a dataframe containing the following columns:
        ['region', 'st_num', 'year', 'month', 'dataset', 'y', 'y_hat']

    Where y_hat is the predictions made using the defined algorithm and input combination.
    """

    if region is None:
        region = 'all_data'

    # Loading predictions
    file_name = f"{algorithm}_{region.lower().replace(' ', '_')}_f{input_combo}.npz"
    pred = np.load(f'models/model_predictions/{file_name}')

    y_hat_train = pred['y_hat_train']
    y_hat_test = pred['y_hat_test']

    # Loading target variable measurements
    df = pd.read_csv('processed_data/et_data.csv')

    if region != 'all_data':
        cond = df['region'] == region
        df = df.loc[cond].reset_index(drop=True)

    # Merging predictions with the measured data
    assert df.shape[0] == y_hat_train.shape[0] + y_hat_test.shape[0], "The predictions do not match the data"

    # Slicing train data
    cond = df['dataset'] == 'train'
    train_data = df.loc[cond]

    # Shuffling train data
    train_data = train_data.sample(frac=1, random_state=12).reset_index(drop=True)
    train_data['y_hat'] = y_hat_train

    # Slicing test data
    cond = df['dataset'] == 'test'
    test_data = df.loc[cond].copy()
    test_data['y_hat'] = y_hat_test

    df = pd.concat([train_data, test_data], ignore_index=True)
    df = df.sort_values(by=['st_num', 'year', 'month']).reset_index(drop=True)

    return df[['region', 'st_num', 'year', 'month', 'dataset', 'ET0', 'y_hat']].rename(columns={'ET0': 'y'})


def load_model_predictions(model, algorithm, region, input_combo):
    '''
    Load regional or general model predictions of the defined algorithm, region, and input combination.

    Parameters:
    -----------

    model: str
        "region" or "general"

    algorithm: str

    region: str

    input_combo: int
    '''

    region_list = ['Black Sea', 'Marmara', 'Central Anatolia', 'Eastern Anatolia',
                   'Aegean', 'Southeastern Anatolia', 'Mediterranean']

    assert model in ['general', 'region'], 'model should be either "general" or "region"'

    if model == 'general':
        df = load_predictions(algorithm, None, input_combo)

        if region is not None and region != 'all_data':
            cond = df['region'] == region
            df = df.loc[cond].reset_index(drop=True)

    else:
        if region is not None and region != 'all_data':
            df = load_predictions(algorithm, region, input_combo)

        else:
            df_list = []
            for region in region_list:
                df_list.append(load_predictions(algorithm, region, input_combo))

            df = pd.concat(df_list, ignore_index=True)

    return df


def load_station_scores(algorithm, region, input_combo):
    """
    Returns the station scores (performance metrics) of the defined algorithm, region, and input_combo model.
    """

    df = pd.read_csv('models/model_results/station_scores.csv')

    model_name = f"{algorithm}_{region.lower().replace(' ', '_')}_f{input_combo}"

    cond = df['model_name'] == model_name

    return df.loc[cond].reset_index(drop=True)


def load_region_general_scores(algorithm, region, input_combo, dataset):
    """
    Returns the general model and regional model scores (performance metrics) of the defined
    algorithm, region, and input combination.

    If an argument is None, all values corresponding to that argument are returned.
    Ex: if input_combo = None, performance metrics for all input combinations are returned
    """

    df = pd.read_csv('models/region_general_scores.csv')

    conds = []

    if algorithm is not None:
        conds.append(df['algorithm'] == algorithm)

    if region is not None:
        conds.append(df['region'] == region)

    if input_combo is not None:
        conds.append(df['input_combo'] == input_combo)

    if dataset is not None:
        conds.append(df['dataset'] == dataset)

    for cond in conds:
        df = df.loc[cond]

    return df.reset_index(drop=True)


def get_train_test_scores(model, algorithm, region=None, input_combo=None, by='region'):
    """
    Returns a table containing the train and test performance metrics of the defined "model" (general or region)
    and "algorithm".

    If "by" is 'region', a value for the argument "input_combo" should be defined and the performance metric
    of all input combinations of the defined region model are returned.

    If "by" is 'input_combo', a value for the argument "region" should be defined and the performance metric
    of all regions of the defined input combination are returned.

    Parameters:
    -----------

    model: str
        "general" or "region"

    algorithm: str

    region: str
        Region name.

    input_combo: int
        Input combination number

    by: str
        "region" or "input_combo"
    """

    # Assertions
    assert by in ['region', 'input_combo'], '"by" should be "region" or "input_combo"'
    assert model in ['general', 'region'], 'model should be either "general" or "region"'

    if by == 'region':
        assert input_combo is not None, 'Define the input combination (input_combo)'
        region = None

    elif by == 'input_combo':
        assert region is not None, 'Define the region'
        input_combo = None

    # Loading model scores
    df = load_region_general_scores(algorithm=algorithm,
                                    region=region,
                                    input_combo=input_combo,
                                    dataset=None)

    # Keeping the chosen model results
    df = df[['metric', 'algorithm', 'region', 'input_combo', 'dataset', f'{model}_metric']]

    df = df.sort_values(by=['metric', 'dataset'], ascending=False)

    # Pivoting table
    df = df.pivot(index=[by], columns=['metric', 'dataset'], values=f'{model}_metric')

    return df


def plot_train_test_scores(metric, algorithm, region=None, input_combo=None, by='region', ax=None):
    """
    Plots a bar graph of the train and test performance metric of the defined "model" (general or region)
    and "algorithm".

    If "by" is 'region', a value for the argument "input_combo" should be defined and the performance metric
    of all input combinations of the defined region model are plotted.

    If "by" is 'input_combo', a value for the argument "region" should be defined and the performance metric
    of all regions of the defined input combination are plotted.

    Parameters:
    -----------

    metric: str
        Model performance metric to be plotted

    model: str
        "general" or "region"

    algorithm: str

    region: str
        Region name.

    input_combo: int
        Input combination number

    by: str
        "region" or "input_combo"
    """

    df = get_train_test_scores('general', algorithm, region, input_combo, by)

    # Setting x-tick label rotation
    if by == 'region':
        rot = 25
    else:
        rot = 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    df[[metric]].plot.bar(color=['gray', 'orange'], ax=ax, rot=rot)

    # Plotting regional model scores
    df = get_train_test_scores('region', algorithm, region, input_combo, by)

    n_models = df.shape[0] * 2

    bars = [bar for bar in ax.get_children() if isinstance(bar, mpl.patches.Rectangle)]
    positions = np.sort([bar.get_x() + bar.get_width() / 2. for bar in bars[:n_models]])

    bar_width_cluster = bars[0].get_width() * 0.22

    # Plotting regional model bars
    for i, dataset in enumerate(['train', 'test']):
        x_coord = positions[i::2]
        ax.bar(x=x_coord, height=df[(metric, dataset)], width=bar_width_cluster, color='black')

    # Annotation

    # Title
    if by == 'input_combo':
        if region == 'all_data':
            region_txt = 'All data'
        else:
            region_txt = region + ' region'

        title = f"{algorithm} / {region_txt}"
    else:
        title = f"{algorithm} / C{input_combo}"

    ax.set_title(title)

    # Axes
    ax.set_xlabel(by.title())
    ax.set_ylabel(metric)
    ax.grid(axis='y')

    # Legend
    legend = ax.legend(labels=['Train', 'Test'])
    handles = legend.get_patches()[:2]
    handles = handles + ax.plot([], [], color='black')
    labels = list(map(lambda x: x.get_text(), legend.get_texts())) + ['Regional Model']

    ax.legend(handles=handles, labels=labels, ncol=3)

    return legend


def get_region_general_scores(dataset, algorithm, region=None, input_combo=None, by='region'):
    """
    Returns a table containing the general and region model performance metrics for the defined dataset and algorithm.

    If "by" is 'region', a value for the argument "input_combo" should be defined and the performance metric
    of all input combinations of the defined region model are returned.

    If "by" is 'input_combo', a value for the argument "region" should be defined and the performance metric
    of all regions of the defined input combination are returned.

    Parameters:
    -----------

    dataset: str
        "train" or "test"

    algorithm: str

    region: str
        Region name.

    input_combo: int
        Input combination number

    by: str
        "region" or "input_combo"
    """

    assert by in ['region', 'input_combo'], '"by" should be "region" or "input_combo"'
    assert dataset in ['train', 'test'], 'dataset should be either "train" or "test"'

    if by == 'region':
        assert input_combo is not None, 'Define the input combination (input_combo)'
        region = None

    elif by == 'input_combo':
        assert region is not None, 'Define the region'
        input_combo = None

    # Loading model scores
    df = load_region_general_scores(algorithm=algorithm,
                                    region=region,
                                    input_combo=input_combo,
                                    dataset=dataset)

    # Pivoting table
    df = df.pivot(index=[by], columns=['metric'], values=['general_metric', 'region_metric'])

    return df


def plot_algorithm_scores(metric, dataset, algorithms, region=None, input_combo=None, by='region', ax=None):
    """
    Plots the general and region model performance metrics of all algorithms for the defined dataset and algorithm.

    If "by" is 'region', a value for the argument "input_combo" should be defined and the performance metric
    of all input combinations of the defined region model are returned.

    If "by" is 'input_combo', a value for the argument "region" should be defined and the performance metric
    of all regions of the defined input combination are returned.

    Parameters:
    -----------

    metric: str
        Model performance metric to be plotted

    dataset: str
        "train" or "test"

    algorithms: list
        List of algorithms to be plotted

    region: str
        Region name.

    input_combo: int
        Input combination number

    by: str
        "region" or "input_combo"
    """

    cond = np.isin(algorithms, ['SVR', 'GPR', 'RF', 'Polynomial'])
    na_algorithm = np.array(algorithms)[~cond]

    assert cond.all(), f"Algorithms {na_algorithm} not found, choose from ['SVR', 'GPR', 'RF', 'Polynomial']"

    color_dict = {'GPR': '#33657d', 'RF': '#c973a6', 'SVR': '#ffb833', 'Polynomial': '#ff826c'}
    colors = [color_dict[algorithm] for algorithm in algorithms]

    # Constructing dataframe for plot
    for i, algorithm in enumerate(algorithms):
        df = get_region_general_scores(dataset, algorithm, region, input_combo, by)

        if i == 0:
            df_general = df[[('general_metric', metric)]]
            df_general = df_general.droplevel(level=0, axis=1)
            df_general = df_general.rename(columns={metric: algorithm})

            df_region = df[[('region_metric', metric)]]
            df_region = df_region.droplevel(level=0, axis=1)
            df_region = df_region.rename(columns={metric: algorithm})

        else:
            df_general[algorithm] = df[[('general_metric', metric)]].copy()
            df_region[algorithm] = df[[('region_metric', metric)]].copy()

    # Setting x-tick label rotation
    if by == 'region':
        rot = 25
    else:
        rot = 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    df_general.plot.bar(color=colors, rot=rot, ax=ax)

    # Getting bar x coordinates
    n_models = len(algorithms) * df.shape[0]

    bars = [bar for bar in ax.get_children() if isinstance(bar, mpl.patches.Rectangle)]
    positions = np.sort([bar.get_x() + bar.get_width() / 2. for bar in bars[:n_models]])

    bar_width_cluster = bars[0].get_width() * 0.22

    # Plotting regional model bars
    for i, algorithm in enumerate(algorithms):
        x_coord = positions[i::len(algorithms)]
        ax.bar(x=x_coord, height=df_region[algorithm], width=bar_width_cluster, color='black')

    # Annotation

    # Title
    if by == 'input_combo':
        if region == 'all_data':
            region_txt = 'All data'
        else:
            region_txt = region + ' region'

        title = f"{region_txt} / {dataset.title()} dataset"
    else:
        title = f"Combination {input_combo} / {dataset.title()} dataset"

    ax.set_title(title)

    # Axes
    ax.set_xlabel(by.replace('_', ' ').title())
    ax.set_ylabel(metric)
    ax.grid(axis='y')

    # Legend
    legend = ax.legend()
    handles = legend.get_patches()[:len(algorithms)]
    handles = handles + ax.plot([], [], color='black')
    labels = list(map(lambda x: x.get_text(), legend.get_texts())) + ['Regional Model']

    ax.legend(handles=handles, labels=labels, ncol=4)


def plot_algorithm_score_change(metric, dataset, algorithms, region=None, input_combo=None, by='region', ax=None):
    """
    Plots the change between general and region model performance metrics for the defined dataset and algorithms.
    The change in metric is computed as a percentage of the general model metric:

        change_in_metric (%) = 100 * (region_metric - general_metric) / general_metric

    If "by" is 'region', a value for the argument "input_combo" should be defined and the change in performance metric
    of all input combinations of the defined region model are returned.

    If "by" is 'input_combo', a value for the argument "region" should be defined and the change in performance metric
    of all regions of the defined input combination are returned.

    Parameters:
    -----------

    metric: str
        Model performance metric to be plotted

    dataset: str
        "train" or "test"

    algorithms: str
        List of algorithms to be plotted

    region: str
        Region name.

    input_combo: int
        Input combination number

    by: str
        "region" or "input_combo"
    """

    cond = np.isin(algorithms, ['SVR', 'GPR', 'RF', 'Polynomial'])
    na_algorithm = np.array(algorithms)[~cond]

    assert cond.all(), f"Algorithms {na_algorithm} not found, choose from ['SVR', 'GPR', 'RF', 'Polynomial']"

    color_dict = {'GPR': '#33657d', 'RF': '#c973a6', 'SVR': '#ffb833', 'Polynomial': '#ff826c'}
    colors = [color_dict[algorithm] for algorithm in algorithms]

    # Constructing dataframe for plot
    for i, algorithm in enumerate(algorithms):
        df_algorithm = get_region_general_scores(dataset, algorithm, region, input_combo, by)

        region_metric = df_algorithm[[('region_metric', metric)]].droplevel(level=0, axis=1)
        general_metric = df_algorithm[[('general_metric', metric)]].droplevel(level=0, axis=1)

        if i == 0:
            df = 100 * (region_metric - general_metric) / general_metric
            df = df.rename(columns={metric: algorithm})

        else:
            df[algorithm] = 100 * (region_metric - general_metric) / general_metric

    # Setting x-tick label rotation
    if by == 'region':
        rot = 25
    else:
        rot = 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    df.plot.bar(color=colors, rot=rot, ax=ax)

    # Annotation

    # Title
    if by == 'input_combo':
        if region == 'all_data':
            region_txt = 'All data'
        else:
            region_txt = region + ' region'

        title = f"{region_txt} / {dataset.title()} dataset"
    else:
        title = f"Combination {input_combo} / {dataset.title()} dataset"

    ax.set_title(title)

    # Axes
    ax.set_xlabel(by.replace('_', ' ').title())
    ax.set_ylabel(metric)
    ax.grid(axis='y')
    ax.axhline(y=0, color='black', linestyle='--', lw=0.5)

    # Legend
    legend = ax.legend(ncol=len(algorithms))


def plot_metric_change_heatmap(metric, dataset, algorithms, region=None, input_combo=None, by='region', ax=None):
    cond = np.isin(algorithms, ['SVR', 'GPR', 'RF', 'Polynomial'])
    na_algorithm = np.array(algorithms)[~cond]

    assert cond.all(), f"Algorithms {na_algorithm} not found, choose from ['SVR', 'GPR', 'RF', 'Polynomial']"

    color_dict = {'GPR': '#33657d', 'RF': '#c973a6', 'SVR': '#ffb833', 'Polynomial': '#ff826c'}
    colors = [color_dict[algorithm] for algorithm in algorithms]

    # Constructing dataframe for plot
    for i, algorithm in enumerate(algorithms):
        df_algorithm = get_region_general_scores(dataset, algorithm, region, input_combo, by)

        region_metric = df_algorithm[[('region_metric', metric)]].droplevel(level=0, axis=1)
        general_metric = df_algorithm[[('general_metric', metric)]].droplevel(level=0, axis=1)

        if i == 0:
            df = (region_metric - general_metric) / general_metric
            df = df.rename(columns={metric: algorithm})

        else:
            df[algorithm] = (region_metric - general_metric) / general_metric

    df = df.T

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(df, ax=ax, annot=True, vmin=-0.1, vmax=+0.1,
                cmap=cmap, fmt='.1%')

    ax.set_ylabel('Algorithm')
    ax.set_xlabel('Combination')

    # Color bar axis
    cax = plt.gcf().axes[-1]
    cax.set_yticklabels(list(map(lambda x: f'{x.get_position()[1]:.0%}', cax.get_yticklabels())))

    # Title
    if by == 'input_combo':
        if region == 'all_data':
            region_txt = 'All data'
        else:
            region_txt = region + ' region'

        title = f"{region_txt} / {dataset.title()} dataset"
    else:
        title = f"Combination {input_combo} / {dataset.title()} dataset"

    ax.set_title(title)

    return ax


def drop_outlier_fun(data, scale=1.5):
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1

    lower_bound = q1 - scale * iqr
    upper_bound = q3 + scale * iqr

    cond = (data > upper_bound) | (data < lower_bound)

    result = data[~cond]
    n_dropped = data.shape[0] - result.shape[0]

    return n_dropped, result


def plot_violin_by_algorithm(algorithms,
                             region,
                             input_combo,
                             dataset='test',
                             ax=None,
                             drop_outliers=True,
                             verbose=False):
    """
    Plots a violin plot of the prediction error of the defined algorithms, region, dataset, and input combination.
    The man and standard deviation of the prediction errors are shown on the plot: Mean (Std)

    Parameters:
    -----------

    algorithms: list
        List of algorithm names to be plotted

    region: str

    input_combo: int

    dataset: str
        "train" or "test"

    drop_outliers: bool
        If True, outliers are dropped to avoid long tails of the violin plot and improve readability.
        The computed mean and std are computed before dropping any outliers.

    verbose: bool
        If True, the number of outliers dropped is printed.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    df_list = []

    # Text offset from the center of the plot
    x_offset1 = 0.07
    x_offset2 = x_offset1 + 0.02
    offset_dict = {'general': -x_offset1, 'region': x_offset2}

    # Grouping data in one dataframe to work with the seaborn violin plot function
    for model in ['general', 'region']:
        for x, algorithm in enumerate(algorithms):

            df = load_model_predictions(model, algorithm, region, input_combo)
            df['algorithm'] = algorithm
            df['error'] = df['y_hat'] - df['y']
            df['Model'] = model.title()

            # Plotting error mean and std
            std = df['error'].std()
            mean = df['error'].mean()

            ax.text(s=f'{mean:.3f} ({std:.3f})', x=x + offset_dict[model], y=0,
                    rotation='vertical', va='center', ha='center', fontsize=12)

            if drop_outliers:
                n_dropped, df['error'] = drop_outlier_fun(df['error'])
                if verbose:
                    print(f'Algorithm: {algorithm} / Input_combo: {input_combo}')
                    print(f'Dropped {n_dropped} outliers from {model} model errors')

            df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Slicing dataset data
    cond = df['dataset'] == dataset
    df = df.loc[cond].reset_index(drop=True)[['algorithm', 'Model', 'error']]

    sns.violinplot(ax=ax, x='algorithm', y='error', hue='Model',
                   split=True, data=df, palette=['skyblue', 'limegreen'])

    # Annotation
    ax.grid(axis='y')
    ax.set_ylabel('Error (Prediction - Measurement)')
    ax.set_xlabel('Algorithm')

    if region == 'all_data' or region is None:
        region_txt = 'All data'
    else:
        region_txt = region + ' region'

    ax.set_title(f'{region_txt} / C{input_combo}')
    ax.legend(ncol=2)

    return df


def plot_violin_by_region(regions,
                          algorithm,
                          input_combo,
                          dataset='test',
                          ax=None,
                          drop_outliers=True,
                          verbose=False):
    """
    Plots a violin plot of the prediction error of the defined regions, algorithm, dataset, and input combination.
    The man and standard deviation of the prediction errors are shown on the plot: Mean (Std)

    Parameters:
    -----------

    regions: list
        List of region names to be plotted

    algorithm: str

    input_combo: int

    dataset: str
        "train" or "test"

    drop_outliers: bool
        If True, outliers are dropped to avoid long tails of the violin plot and improve readability.
        The computed mean and std are computed before dropping any outliers.

    verbose: bool
        If True, the number of outliers dropped is printed.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    df_list = []

    # Text offset from the center of the plot
    x_offset1 = 0.07
    x_offset2 = x_offset1 + 0.02
    offset_dict = {'general': -x_offset1, 'region': x_offset2}

    # Grouping data in one dataframe to work with the seaborn violin plot function
    for model in ['general', 'region']:
        for x, region in enumerate(regions):

            df = load_model_predictions(model, algorithm, region, input_combo)
            df['error'] = df['y_hat'] - df['y']
            df['Model'] = model.title()

            # Plotting error mean and std
            std = df['error'].std()
            mean = df['error'].mean()

            ax.text(s=f'{mean:.3f} ({std:.3f})', x=x + offset_dict[model], y=0,
                    rotation='vertical', va='center', ha='center', fontsize=12)

            if drop_outliers:
                n_dropped, df['error'] = drop_outlier_fun(df['error'])
                if verbose:
                    print(f'Algorithm: {algorithm} / Input_combo: {input_combo}')
                    print(f'Dropped {n_dropped} outliers from {model} model errors')

            df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Slicing dataset data
    cond = df['dataset'] == dataset
    df = df.loc[cond].reset_index(drop=True)[['region', 'Model', 'error']]

    sns.violinplot(ax=ax, x='region', y='error', hue='Model',
                   split=True, data=df, palette=['skyblue', 'limegreen'])

    # Annotation
    ax.grid(axis='y')
    ax.set_ylabel('Error (Prediction - Measurement)')
    ax.set_xlabel('Region')

    ax.set_title(f'{algorithm} / C{input_combo}')
    ax.legend(ncol=2)

    return df


def plot_station_series(model, algorithm, input_combo, dataset='test', region=None, st_num=None, ax=None):
    """
    Plots the measured and estimated ET0 time series for a station.

    If "st_num" is defined, "region" is ignored and this station's time series is plotted.

    If "st_num" is None and "region" is defined, a random station from this region's "dataset" is plotted.

    Parameters:
    -----------

    model: str
        "general" or "region"

    algorithm: str

    input_combo: int

    dataset: str
        "train" or "test"

    region: str

    st_num: int
        Station number (ID) to be plotted

    """
    df = load_model_predictions(model, algorithm, 'all_data', input_combo)
    # Slicing test or train dataset
    cond = df['dataset'] == dataset
    df = df.loc[cond]

    if st_num is not None:
        assert st_num in df['st_num'].unique(), f"Station {st_num} not found in the {dataset} dataset"
        cond = df['st_num'] == st_num
        df = df.loc[cond].sort_values(by=['year', 'month']).reset_index(drop=True)
        region = df['region'].unique()[0]

    elif region is not None:
        assert region in df['region'].unique(), "Region {region} is not defined in the dataset"

        # Sampling a random station from the defined region
        cond = df['region'] == region
        df = df.loc[cond]

        # Sampling random station from region
        st_num = np.random.choice(df['st_num'].unique())

        # Slicing the sampled station
        cond = df['st_num'] == st_num
        df = df.loc[cond].sort_values(by=['year', 'month']).reset_index(drop=True)

    else:
        raise ValueError("Define either 'region' or 'st_num'")

    # Adding date column
    df['day'] = 15
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.rename(columns={'y': 'ET0'})  # To avoid warning for y column name in plot function
    # Plotting time series
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot('date', 'ET0', label='Measured ET0', linestyle='--', color='green',
            linewidth=1.5, marker='o', markersize=5, data=df)

    ax.plot('date', 'y_hat', label='Estimated ET0', linestyle='--', color='skyblue',
            linewidth=1.5, marker='^', markersize=5, data=df)

    ax.set_xlabel('Date')
    ax.set_ylabel('ET0 (mm/day)')
    title = f"{region} region / Station {st_num} ({dataset}) / {algorithm} algorithm / {model.title()} model / C{input_combo}"
    ax.set_title(title)
    ax.legend(ncol=2)