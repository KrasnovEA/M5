import json
import numpy as np
import pandas as pd

import sys
import argparse


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-m', '--min_train_day', default=1180, type=int)
 
    return parser



def transform_day_to_int(str_day):
    """
    
    Перевод дня в численное представление

    Parameters
    ----------
    str_day : string
        Строка, описывающая день в исходных данных

    Returns
    -------
    int
        Численное предстваление дня
    """    
    return int(str_day.rpartition('_')[2])

def get_label_encoding_(df, columns):
    """
    
    Label-encoding категориальныз признаков

    Parameters
    ----------
    df : DataFrame
        Датафрейм с признаками
    columns : list
        Список колонок для кодирования

    Returns
    -------
    dict
        Словарь кодирования категориальных признаков: {'column_name': {'label': code}}
    """    
    label_codes = {}
    for column in columns:
        df[column] = df[column].astype('category')
        column_dict = dict(zip(df[column], df[column].cat.codes) )
        df[column] = df[column].cat.codes
        label_codes[column] = column_dict
    return label_codes

def melt_and_merge_sales_df(sales_df, dates_df, prices_df,  add_days_qty=28, min_train_day=1180):
    """
    
    Объединение и реструктуризация исходных датафреймов

    Parameters
    ----------
    sales_df : DataFrame
        Данные о продажах
    dates_df : DataFrame
        Данные о календаре
    prices_df : DataFrame
        Данные о ценах
    add_days_qty : int, optional
        Число дней, на которое надо расширить исходный датафрейм (пустыми значениями),
        by default 28
    min_train_day : int, optional
        Первый день, с которого начинаются данные для обучения, by default 1180

    Returns
    -------
    DataFrame, dict
        Объединенный датафрейм с признаками и словарь кодирования категориальных признаков
    """    
    id_variables = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    last_train_day = transform_day_to_int(sales_df.columns[-1])
    pred_cols = [f'd_{i}' for i in range(last_train_day + 1, last_train_day + 1 + add_days_qty)]
    for col in pred_cols:
        sales_df[col] = np.nan
    sales_df = pd.melt(
        sales_df,
        id_vars=id_variables,
        var_name='day',
        value_name='demand'
    )
    sales_df['day_as_int'] = sales_df['day'].apply(transform_day_to_int)
    sales_df = sales_df[sales_df['day_as_int'] >= min_train_day]
    sales_df = sales_df.merge(dates_df, on=['day'], copy=False, how='left')
    # после этого мерджа пропадут дни, когда товар не выставлялся на продажу
    sales_df = sales_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], copy=False, how='inner')
    date_variables = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    label_codes_dict = get_label_encoding_(sales_df, columns=id_variables + date_variables)
    sales_df.drop(['day', 'wm_yr_wk', 'weekday', 'date'], axis=1, inplace=True)
    return sales_df, label_codes_dict


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    min_train_day = namespace.min_train_day

    dates_df = pd.read_csv('data/calendar.csv')
    sales_df = pd.read_csv('data/sales_train_validation.csv')
    prices_df = pd.read_csv('data/sell_prices.csv')

    dates_df.rename(columns={'d': 'day'}, inplace=True)
    sales_df, label_codes_dict = melt_and_merge_sales_df(
        sales_df, dates_df, prices_df, min_train_day=min_train_day
    )

    sales_df.to_csv('data/melted_train.csv')
    with open('data/label_codes_dict.json', 'w') as f:
        json.dump(label_codes_dict, f)
