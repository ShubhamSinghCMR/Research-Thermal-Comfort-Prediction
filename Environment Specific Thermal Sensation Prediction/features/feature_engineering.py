import pandas as pd

def get_all_sheet_names(path):
    x = pd.ExcelFile(path)
    return x.sheet_names

def load_environment_sheet(sheet_name, path='dataset/input_dataset.xlsx'):
    return pd.read_excel(path, sheet_name=sheet_name)
