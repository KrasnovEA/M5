import pandas as pd
import argparse
import sys
import json
import re
import numpy as np

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-c', '--config', type=str)
 
    return parser


def generate_past_demand_features(
        feature_df,
        shift_parameters,
        mean_parameters,
        mean_period_parameters,
        exp_mean_parameters,
        exp_mean_period_parameters,
):
    """
    
    Генерация признаков на основе предыдущих продаж

    Parameters
    ----------
    feature_df : DataFrame
        Датафрейм, полученный после скрипта generate_melted_df.py
    shift_parameters : list of tuples
    mean_parameters : list of tuples
    mean_period_parameters : list of tuples
    exp_mean_parameters : list of tuples
    exp_mean_period_parameters : list of tuples

    Examples
    --------
    Например для набора признаков
        f(t, 7, 1, 1), f(t, 28, 1, 1), f(t, 7, 7, 1), f(t, 28, 7, 1),
        f(t, 7, 7, 7), f(t, 28, 7, 7), g(t, 7, 0.5, 1), g(t, 7, 0.5, 7)
    аргументы будут следующими:
        shift_parameters=[7, 28]
        mean_parameters=[(7, 7), (28, 7)]
        mean_period_parameters=[(7, 7, 7), (28, 7, 7)]
        exp_mean_parameters=[(7, 0.5)]
        exp_mean_period_parameters=[(7, 0.5, 7)]
    
    Returns
    -------
    DataFrame
        Датафрейм с рассчитанными признаками
    """    
    for shift_value in shift_parameters:
        column_name = f'shift_{shift_value}'
        feature_df[column_name] = feature_df.demand.shift(shift_value, fill_value=0)
    
    for shift_value, window_size in mean_parameters:
        column_name = f'mean_{shift_value}_{window_size}'
        feature_df[column_name] = feature_df.demand.shift(shift_value, fill_value=0).rolling(window_size).mean()
    
    for shift_value, window_size, period_size in mean_period_parameters:
        def f(x):
            ind = np.argwhere(np.isnan(x))
            if ind.shape[0] != 0:
                x = x[np.max(ind):]
            return x[::-period_size].mean()
        
        column_name = f'mean_{shift_value}_{window_size}_period_{period_size}'
        feature_df[column_name] = feature_df.demand.shift(
            shift_value, fill_value=0
        ).rolling(window_size * period_size).apply(f, raw=True).values
    
    for shift_value, b in exp_mean_parameters:
        def f(x):
            ind = np.argwhere(np.isnan(x))
            if ind.shape[0] != 0:
                x = x[np.max(ind):]
            beta = b ** np.arange(len(x))[::-1]
            return np.sum(beta * x) / np.sum(beta)
    
        column_name = f'mean_{shift_value}_{b:.3f}'
        feature_df[column_name] = feature_df.demand.shift(shift_value, fill_value=0).rolling(20, min_periods=1).apply(f, raw=True).values
    
    for shift_value, b, period_size in exp_mean_period_parameters:
        def f(x):
            ind = np.argwhere(np.isnan(x))
            if ind.shape[0] != 0:
                x = x[np.max(ind):]
            x = x[::-period_size]
            beta = b ** np.arange(len(x))
            return np.sum(beta * x) / np.sum(beta)
        column_name = f'mean_{shift_value}_{b:.3f}_period_{period_size}'
        feature_df[column_name] = feature_df.demand.shift(shift_value, fill_value=0).rolling(20*period_size, min_periods=1).apply(f, raw=True).values
    
    
    return feature_df
    
    
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    config_string = re.sub('%', '"', namespace.config)
    config = json.loads(config_string)    

    id_columns = ['id']
    
    feature_df = pd.read_csv(
        'data/melted_train.csv',
        dtype={one_id_column: "category" for one_id_column in id_columns},
        usecols=id_columns + ['demand'],
    )
    
    feature_df = generate_past_demand_features(
        feature_df,
        **config,
    )
    
    # drop target
    feature_df.drop(['demand'], axis=1, inplace=True)
    
    feature_df.to_csv('data/past_demand_features.csv', index=False)