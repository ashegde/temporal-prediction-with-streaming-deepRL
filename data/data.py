"""
This module provides functionality for downloading the ETTm2.csv dataset. 
More details on this dataset can be found in the reference below,

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021, May).
Informer: Beyond efficient transformer for long sequence time-series forecasting.
In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 12, pp. 11106-11115).

or at the following github link: https://github.com/zhouhaoyi/ETDataset
"""

import os
import requests
import pandas as pd

def download_data(save_file: str) -> None:
    """
    Downloads the ETTm2.csv dataset from the specified URL.

    Args:
    save_file (str): path to where the data will be downloaded
    """

    url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv'
    content = requests.get(url).content
    with open(save_file, 'wb') as f:
        f.write(content)


def load_data() -> pd.DataFrame:
    """
    Loads the ETTm2 dataset.

    Returns:
    (pd.DataFrame): the ETTm2 dataset 
    """
      
    save_file = 'data/ETTm2.csv'
    if not os.path.exists(save_file):
        download_data(save_file)
    
    return pd.read_csv(save_file)


def prepare_data(skip: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the ETTm2 dataset and splits it into feature
    and target dataframes.


    Returns:
    (pd.DataFrame): containing features available ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    (pd.DataFrame): containing target ['OT'] 
    """
    df = load_data()
    df['OTprev'] = df['OT']
    cnames = df.columns.tolist() 
    cnames[-2], cnames[-1] = cnames[-1], cnames[-2] # swap OT and OTprev
    df = df.reindex(columns=cnames)

    # shift OTprev forward, previous time step OT becomes available for input data at next step
    df.OTprev = df.OTprev.shift(1) 
    df.drop([0], inplace=True)
    
    if skip > 0:
        df = df.iloc[::skip]


    inputs = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OTprev']]
    targets = df['OT']
    
    return inputs, targets