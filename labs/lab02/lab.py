# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    tricky_1 = pd.DataFrame({
        'Name': [1, 2, 3, 4, 5],
        'Name': [6, 7, 8, 9, 10],
        'Age': [11, 12, 13, 14, 15]
    })

    tricky_1.to_csv('tricky_1.csv', index=False)

    tricky_2 = pd.read_csv('tricky_1.csv')

    del tricky_2

    # Observation: The data with the first 'Name' column is overriden by the second one
    return 3


def trick_bool():
    bools = pd.DataFrame({
        True: [1, 2, 3, 4],
        True: [5, 6, 7, 8],
        False: [9, 10, 11, 12],
        False: [13, 14, 15, 16]
    })

    # bools[True]
    # It works the same as last func where it override with second column
    # It returns as a series as I am expected, but there is no such choices

    # bools[[True, True, False, False]]
    # The columns are interpreted as binary masking array to filter out rows by True value
    # I didn't expect that because I did not recognize the binary masking
    # So that returns 2 rows
    # It indexed the columns using dictionary, which has to be unique
    # So it only returns 2 columns

    # bools[[True, False]]
    # It interpreted the columns again as binary masking
    # Which expect the len(mask) == len(rows), but this is not the case
    # Therefore, it throws an error

    return [4, 12, 4]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    # Non-Null count/proportion
    not_na_binary_mask = df.notna()
    num_nonnull = not_na_binary_mask.sum(axis=0)
    prop_nonnull = not_na_binary_mask.mean(axis=0)

    # Non-Null unique count/proportion
    not_na_unique = df[not_na_binary_mask]
    num_distinct = not_na_unique.nunique()
    prop_distinct = num_distinct / not_na_unique.shape[0]

    # Create new dataframe
    return pd.DataFrame(
        {
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
        },
        index=df.columns
    )


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    ...

def university_info(cleaned):
    ...

