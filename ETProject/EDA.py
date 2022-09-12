import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_station_count_by_region(st_data, et_data, by='region', figsize=(18, 4)):
    '''
    Plot the number of stations and datapoints per region or province.

    '''
    colors = ['skyblue', 'limegreen']
    linewidth = 4

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle(f'{by.title()} Details', fontsize=16)

    st_df = st_data.copy()
    et_df = et_data.copy()

    if by == 'province':
        et_df = et_df.merge(st_df[['st_num', 'province']], on='st_num')

    # Computing station count
    region_data = st_df.groupby(by)[['st_num']].count().reset_index().rename(columns={'st_num': 'station_count'})
    # Computing data count
    data_count = et_df.groupby(by)[['st_num']].count().reset_index().rename(columns={'st_num': 'data_count'})
    region_data = region_data.merge(data_count, on=by)

    # Plotting
    colors = ['skyblue', 'limegreen']
    for i, variable in enumerate(['data_count', 'station_count']):
        region_data = region_data.sort_values(by=variable)
        ax = axs[i]
        ax.barh(y=by, width=variable, data=region_data, color=colors[i])
        ax.set_xlabel(variable.replace('_', ' ').title())
        ax.set_title(variable.replace('_', ' ').title())
        ax.set_ylim(-1, region_data.shape[0])
        ax.grid(axis='x')


def transform(data):
    '''
    Transform the given data and return the transformed data and a function to untransform data.

    Used to compute marker sizes on map in the function plot_station_map
    '''

    # Modify these variable to change the marker size in general and variance between different sizes
    a = 2
    b = 0.5
    c = 40

    trans_data = (data - data.min()) / (data.max() - data.min())
    trans_data = (a * trans_data + b) * c

    def untransform(data, a=a, b=b, c=c, min_val=data.min(), max_val=data.max()):
        orig_data = (data / c - b) / a
        orig_data = orig_data * (max_val - min_val) + min_val

        return orig_data

    return trans_data, untransform


def plot_station_map(st_data, et_data, region_geo):
    """
    Plots the station on the map, each station sized by the number of datapoints it has and colored by its region.
    """
    # Computing data per station
    st_df = st_data.copy()
    data_count = et_data.groupby('st_num')[['ET0']].count().reset_index().rename(columns={'ET0': 'data_count'})
    st_df = st_df.merge(data_count, on='st_num')

    # Adding color column
    region_colors = {
        'Mediterranean': '#E66E37',
        'Central Anatolia': '#E6E345',
        'Marmara': '#AF77E6',
        'Black Sea': '#7CE678',
        'Southeastern Anatolia': '#E6A182',
        'Eastern Anatolia': '#E4E67E',
        'Aegean': '#6CE6D9'}
    st_df['color'] = st_df['region'].replace(region_colors)

    # Compute the scatter marker sizes and get the untransform function
    scatter_size, untransform_fun = transform(st_df['data_count'])

    fig, ax = plt.subplots(figsize=(12, 6))
    region_geo.explode(index_parts=False).exterior.plot(ax=ax, color='black', linewidth=0.6)

    sc = ax.scatter(x='longitude', y='latitude', s=scatter_size,
                    data=st_df, color='color', edgecolor='black')

    # Annotation
    ax.set_title('Data count per station')

    _, msizes = np.histogram(scatter_size, bins=3)
    labels = untransform_fun(msizes)

    markers = []
    for size, label in zip(msizes, labels):
        markers.append(plt.scatter([], [], s=size, color='w', edgecolor='black',
                                   label=f'{label:.0f}'))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(np.arange(25, 46))

    ax.legend(handles=markers, title='Number of datapoints', ncol=4, loc=4)

    region_name_coord = {
        'Marmara': (27.5, 39.8),
        'Aegean': (28.4, 38.6),
        'Mediterranean': (29.3, 37.03),
        'Central Anatolia': (34.5, 39),
        'Black Sea': (34.5, 41.2),
        'Eastern Anatolia': (39, 39),
        'Southeastern Anatolia': (38, 37)
    }

    for region_name, coord in region_name_coord.items():
        ax.text(x=coord[0], y=coord[1], s=region_name)


def plot_region_KDE(ax,
                    et_data,
                    variable,
                    region_name,
                    bw=0.2):
    """
    Plots the the test and train data KDE of the defined variable using the defined region data,
    along with the KDE of the variable using the whole dataset.

    If region_name is None, the test and train data distributions of the entire dataset are plotted.
    """

    colors = ['black', '#1c6fff']

    alpha = 0.5

    if region_name is not None:
        cond = et_data['region'] == region_name
        region_data = et_data.loc[cond]
    else:
        region_data = et_data

    for dataset, linestyle in zip(['train', 'test'], [None, '--']):
        cond = region_data['dataset'] == dataset
        graph_data = region_data.loc[cond, variable].to_numpy()
        sns.kdeplot(graph_data, bw_method=bw, ax=ax, label=f'Region {dataset}', color=colors[1], linestyle=linestyle)

    # Plotting all Data
    if region_name is not None:
        all_data = et_data.loc[:, variable].to_numpy()
        sns.kdeplot(all_data, bw_method=bw, ax=ax, label=f'All data', color=colors[0])

    ax.set_title(variable, fontsize=12)
    ax.set_ylabel(None)


def plot_station_timeseries(et_data, st_num, variables):
    """
    Plots one or two variable time series of the defined station from et_data using separate y axes.

    Parameters:
    -----------

    variables: list
        A list of one or two column names (variables) from et_data.
    """
    colors = ['skyblue', 'darkorange']

    assert len(variables) <= 2, "Choose one or two variables only"

    cond = et_data['st_num'] == st_num
    graph_data = et_data.loc[cond, ['year', 'month'] + variables].copy()

    graph_data['day'] = 15
    graph_data['date'] = pd.to_datetime(graph_data[['year', 'month', 'day']])

    fig, ax = plt.subplots(figsize=(18, 5))

    # Handles to create a legend
    handles = []

    for i, variable in enumerate(variables):
        if i == 1:
            ax = ax.twinx()

        ax.plot('date', variable, data=graph_data,
                color=colors[i], marker='o', ms=4)
        ax.set_ylabel(variable)
        handles.append(ax.get_lines()[0])

    ax.set_xlabel('Date')
    ax.set_title(f'Station: {st_num}')
    ax.legend(handles=handles, labels=variables, ncol=2)

    return ax