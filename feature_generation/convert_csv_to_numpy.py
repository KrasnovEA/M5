import pandas as pd
import numpy as np
import json


def count_lines(file_path='data/past_demand_features.csv'):
    """
    
    Подсчет количества строк в csv файле

    Parameters
    ----------
    file_path : str, optional
        Путь к файлу, by default 'data/past_demand_features.csv'

    Returns
    -------
    int
        Количество линий в файле
    """    
    line_amount = 0
    with open(file_path) as f:
        for i, _ in enumerate(f):
            if i == 0:
                continue
            line_amount += 1
    return line_amount


def create_numpy_array_from_csv(
    category_df_batch_iterator, past_demands_df_batch_iterator, line_amount
):
    """
    
    Перевод DataFrame в numpy array

    Parameters
    ----------
    category_df_batch_iterator : iterator
        Итератор, полученный при чтении csv с флагом iterator=True
    past_demands_df_batch_iterator : iterator
        Итератор, полученный при чтении csv с флагом iterator=True
    line_amount : int
        Количество линий, полученное после вызова count_lines

    Returns
    -------
    numpy ndarray, list
        признаки в формате np.array и список названий колонок
    """    
    zip_iterator = zip(category_df_batch_iterator, past_demands_df_batch_iterator)
    for i, one_batch in enumerate(zip_iterator):
        one_batch[0].drop(['id'], axis=1, inplace=True)
        one_batch[1].drop(['id'], axis=1, inplace=True)
        one_batch[1].fillna(-1, inplace=True)

        if i == 0:
            feature_array_shape = one_batch[0].shape[1] + one_batch[1].shape[1]
            header = one_batch[0].columns.tolist() + one_batch[1].columns.tolist()
            final_feature_array = np.zeros((line_amount, feature_array_shape))

        final_feature_array[i * batch_size:(i + 1) * batch_size] = (
            np.hstack((one_batch[0].values, one_batch[1].values))
        )
    
    return final_feature_array, header


if __name__ == '__main__':
    batch_size = 1000000
    
    category_df_batch_iterator = pd.read_csv(
        'data/melted_train.csv',
        iterator=True,
        chunksize=batch_size,
        index_col=0,
    )
    
    past_demands_df_batch_iterator = pd.read_csv(
        'data/past_demand_features.csv',
        iterator=True,
        chunksize=batch_size,
    )
    
    line_amount = count_lines()        
    final_feature_array, header = create_numpy_array_from_csv(
        category_df_batch_iterator,
        past_demands_df_batch_iterator,
        line_amount,
    )
    
    day_index = header.index('day_as_int')
    
    for main_val_type in ['validation', 'evaluation']:
        border_elements = dict()
        if main_val_type == 'validation':
            possible_val_types = ['train', 'val', 'test']
            border_elements['train'] = (1210, max(final_feature_array[:, day_index]) - 28 * 3)
            border_elements['val'] = (border_elements['train'][1] + 1, border_elements['train'][1] + 28)
            border_elements['test'] = (border_elements['val'][1] + 1, border_elements['val'][1] + 28)
        elif main_val_type == 'evaluation':
            possible_val_types = ['train', 'test']
            border_elements['train'] = (1210, max(final_feature_array[:, day_index]) - 28)
            border_elements['test'] = (border_elements['train'][1] + 1, border_elements['train'][1] + 28)

        for val_type in possible_val_types:
            X_val_type = final_feature_array[
                (final_feature_array[:, day_index] >= border_elements[val_type][0]) & 
                (final_feature_array[:, day_index] <= border_elements[val_type][1])
            ]
            np.save(f'data/X_{val_type}_{main_val_type}', X_val_type)
    
    with open('data/final_feature_header.json', 'w') as f:
        json.dump(header, f)