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

def concatenate_series(ser, max_length):
    extra_entries = pd.Series([np.nan] * (max_length - len(ser)))
    ser = (
        pd.concat(
            [ser, extra_entries],
            ignore_index=True
        )
    )
    return ser

def most_common(df, N=10):
    # Extract columns name
    original_columns = df.columns

    # Iterate columns
    for col in original_columns:
        # Compute original frequency and its values
        freq = df[col].value_counts()
        most_freq_vals = pd.Series(freq.index)

        # Extend freq/values to align with df shape
        if freq.shape[0] < N: freq = concatenate_series(freq, N)
        if len(most_freq_vals) < N: most_freq_vals = concatenate_series(most_freq_vals, N)

        # Add that to df
        df = (
            df[df.index < N]
            .assign(**{
                col+'_values': most_freq_vals,
                col+'_counts': freq
            })            
        )
    df = df.drop(columns=original_columns)
    return df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    ## The name of the superhero with the greatest number of superpowers
    # Find num of power each hero has
    num_powers = (
        powers
        .iloc[:, 1:]
        .sum(axis=1)
    )
    # Find the index of the most powerful hero
    greatest_power_index = (
        np.argsort(num_powers)
        .iloc[-1]
    )
    # Find the hero name based on the index
    powerful_hero = (
        powers
        .loc[greatest_power_index, 'hero_names']
    )

    ## Identify the most common superpower among superheroes who can fly, other than 'Flight' itself
    most_common_power = (
        powers[powers['Flight'] == True]
        .drop(columns='Flight')
        .iloc[:, 1:]
        .sum(axis=0)
        .sort_values(ascending=False)
        .index[0]
    )

    ## The name of the most common superpower among superheroes with only one superpower
    most_common_one_power = (
        powers[num_powers == 1]
        .iloc[:, 1:]
        .idxmax(axis=1)
        .value_counts()
        .index[0]
    )

    return [
        powerful_hero, 
        most_common_power, 
        most_common_one_power
    ]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace("-", np.nan).map(lambda x: np.nan if isinstance(x, float) and (x < 0) else x)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return [
        'Onslaught', 
        'DC Comics', 
        'bad', 
        'Marvel Comics', 
        'NBC - Heroes', 
        'Groot'
    ]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    # Replace new line with ', ' for institution name
    inst_with_new_line_mask = df['institution'].str.contains('\n')
    df.loc[inst_with_new_line_mask, 'institution'] = (
        df.loc[inst_with_new_line_mask, 'institution']
        .str
        .replace('\n', ', ')
    )

    # Change data type for 'broad impact'
    df['broad_impact'] = df['broad_impact'].astype(np.int32)


    # Split tuple from 'national_rank' into two part
    df = df.assign(**{
        'nation': df['national_rank']
            .transform(lambda x: x.split(',')[0]),
        'national_rank_cleaned': df['national_rank']
            .transform(lambda x: x.split(',')[1])
            .astype(np.int32)
    })
    df = df.drop(columns='national_rank')

    # Replace some country names
    df['nation'] = (
        df['nation']
        .replace(to_replace='Czechia', value='Czech Republic')
        .replace(to_replace='USA', value='United States')
        .replace(to_replace='UK', value='United Kingdom')
    )

    # Detect r1 public school
    df = df.assign(**{
        'is_r1_public': np.where(
            (df['control'] == 'Public') & 
            (df['city'].notna()) & 
            (df['state'].notna()), 
            True, 
            False)
    })

    return df

def university_info(cleaned):
    # Prompt 1: State's with >= 3 inst, States has lowest mean score
    over_3_inst = (
        cleaned
        .groupby(by='state')
        ['state']
        .filter(lambda x: len(x) >= 3)
        .unique()
    )
    min_state = (
        cleaned[cleaned['state'].isin(over_3_inst)]
        .groupby(by='state')
        ['score']
        .aggregate(['mean'])
        .idxmin()
        .item()
    )

    # Prompt 2: Base is School within Rank 100, find proportion of school with Quality of Faculty within 100
    rank100 = cleaned[cleaned['world_rank'] <= 100].shape[0]
    quality100 = cleaned[cleaned['world_rank'] <= 100][cleaned['quality_of_faculty'] <= 100]['quality_of_faculty'].shape[0]
    proportion100 = quality100 / rank100

    # Prompt 3: Find states with over 50% of private schools
    all_shool_counts = (
        cleaned
        .groupby('state')
        ['control']
        .aggregate(['size'])
        .sort_values(by='state')
        ['size']
    )
    all_school_states = all_shool_counts.index

    private_school_counts = (
        cleaned[cleaned['control'] != 'Public']
        .groupby('state')
        ['control']
        .aggregate(['size'])
    )
    # States without private would not appear here, for later division to be successful
    # I need to populate state with 0 private schools
    non_private_states = set(all_school_states) - set(private_school_counts.index)
    extra_private_school_counts = pd.DataFrame(
        {'size': [0] * len(non_private_states)}, 
        index=list(non_private_states)
    )
    private_school_counts = (
        pd.concat([private_school_counts, extra_private_school_counts])
        .sort_index()
        ['size']
    )
    num_states_over_50_percent = (
        private_school_counts[
            (private_school_counts / all_shool_counts) >= 0.5
        ]
        .index.shape[0]
    )

    # Prompt 4: For the best inst in each country, find the lowest rank in the world
    best_inst_per_nation = (
        cleaned.groupby(by='nation')
        ['national_rank_cleaned']
        .idxmin()
        .values
    )

    lowest_rank_best_inst_nation = (
        cleaned
        .iloc[best_inst_per_nation, :]
        .sort_values(by='world_rank', ascending=False)
        .iloc[0]
        ['institution']
    )

    return [
        min_state,
        proportion100,
        num_states_over_50_percent,
        lowest_rank_best_inst_nation
    ]
