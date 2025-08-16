# utils.py
import pandas as pd
import numpy as np

def load_data(target_col=None, log_transform=False):
    """
    Загружает данные из cleaned_data.csv и возвращает признаки X и целевую переменную y.
    
    Параметры:
    - target_col: название столбца (например, 'IC50, mM')
    - log_transform: применить ли log10 к целевой переменной
    """
    # Чтение данных
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Столбцы, которые точно не должны быть в X
    columns_to_exclude = [
        'IC50, mM', 'CC50, mM', 'SI',
        'log_IC50', 'log_CC50', 'log_SI',
        'Profile', 'is_promising', 'is_efficient', 'is_safe'
    ]
    
    # Удаляем только те, что есть в данных
    existing_cols_to_drop = [col for col in columns_to_exclude if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    
    if target_col is None:
        return X  # возвращаем только признаки
    
    y = df[target_col]
    if log_transform:
        y = np.log10(y + 1e-6)  # защита от log(0)
    
    return X, y