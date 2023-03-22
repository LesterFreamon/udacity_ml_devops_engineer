"""This module contains functions for importing data from csv files"""
from typing import IO

import pandas as pd


def import_data(file_obj: IO[str]) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at file_obj

    input:
        file_obj: a file object for the csv
    output:
        df: pandas dataframe
    '''
    df = pd.read_csv(file_obj, index_col=0)
    return df
