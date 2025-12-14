import os
import pandas as pd
from typing import Union
from pathlib import Path
import streamlit as st

def detect_data_type(file_path: Union[str, Path]) -> str:
    file_path = str(file_path).lower()
    
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        return 'image'
    elif file_path.endswith('.csv'):
        return 'tabular'
    else:
        return 'unknown'

def detect_task_type(data: pd.DataFrame) -> str:
    if data.empty:
        raise ValueError("DataFrame boş olamaz")
        
    target = data.columns[-1]
    unique_values = data[target].nunique()
    
    if pd.api.types.is_numeric_dtype(data[target]):
        if unique_values <= 10:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'

def check_directory_structure(directory: str) -> bool:
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not subdirs:
        return False
    
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        for file in os.listdir(subdir_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                return True
    
    return False
