# lab.py


import os
import io
import re
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def preprocess_df(fp):
    # Define renaming rules
    rename_rules = {
        'CURRENT': 'CURRENT_COMPANY',
        'JOB': 'JOB_TITLE',
        'FIRST': 'FIRST_NAME',
        'LAST': 'LAST_NAME',
        'EMAIL': 'EMAIL',
        'UNIVERSITY': 'UNIVERSITY'
    }
    split_name_pat = r'_|\s'

    return (
        # Read rest of the dfs
        pd.read_csv(fp)
        # Rename col based on rules
        .rename(
            mapper=lambda x: rename_rules[re.split(split_name_pat, x)[0].upper()], 
            inplace=False,
            axis=1,
            errors='raise'
        )
        # Re-order columns
        [['CURRENT_COMPANY', 'JOB_TITLE','FIRST_NAME', 'LAST_NAME', 'EMAIL', 'UNIVERSITY']]
    )

def read_linkedin_survey(dirname):
    # Convert to Path object
    dirp = Path(dirname)

    # Intialize df
    df = None

    # Iterate each CSV files and vstacking df together
    for i, fp in enumerate(dirp.iterdir()):
        # Read the first df
        if not i: 
            df = preprocess_df(fp)

        else:
            # Concatenate the rest dfs to original df
            df = pd.concat(
                objs=[df, preprocess_df(fp)],
                axis=0,
                ignore_index=True,
                copy=True
            )
    
    return df

def com_stats(df):
    # The proportion of people who went to a university with the string
    #  'Ohio' in its name that have the string 'Programmer' somewhere in their job title.
    ohio_mask = df['UNIVERSITY'].str.contains(
        pat='ohio', 
        case=False, 
        na=False, 
        regex=False
    )
    programmer_mask = df['JOB_TITLE'].str.contains(
        pat='programmer', 
        case=False, 
        na=False, 
        regex=False
    )
    prompt_1 = df.loc[ohio_mask & programmer_mask, :].shape[0] / df.shape[0]

    # The number of job titles that **end** with the exact string `'Engineer'`. 
    # Note that we're asking for the number of job titles, **not** the number of people!
    unique_names = df['JOB_TITLE'].unique()
    prompt_2 = len([
        name 
        for name in unique_names 
        if (isinstance(name, str)) and (name.endswith('Engineer'))
    ])

    # The job title that has the longest name (there are no ties)
    valid_names = [name for name in unique_names if isinstance(name, str)]
    prompt_3 = sorted(valid_names, key=lambda x: len(x), reverse=True)[0]

    # The number of people who have the word `'manager'` in their job title, 
    # uppercase or lowercase (`'Manager'`, `'manager'`, and `'mANAgeR'` should all count).
    manager_mask = df['JOB_TITLE'].str.contains(
        pat='manager', 
        case=False, 
        na=False, 
        regex=False
    )
    prompt_4 = df.loc[manager_mask, :].shape[0]

    return [prompt_1, prompt_2, prompt_3, prompt_4]



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    # Convert to Path object
    dirp = Path(dirname)

    # Intialize df
    df = None

    # Define rename rules
    rename_rule = {
        i : col 
        for i, col in enumerate(['name', 'movie', 'genre', 'animal', 'plant', 'color'])
    }

    # Iterate each CSV files and hstacking df together
    for i, fp in enumerate(sorted(list(dirp.iterdir()))):
        # Read the first df
        if not i: 
            df = pd.read_csv(fp).drop(columns=['id'])

        else:
            # Concatenate the rest dfs to original df
            df = pd.concat(
                objs=[df, pd.read_csv(fp).drop(columns=['id'])],
                axis=1,
                join='outer',
                ignore_index=True,
                copy=True
            )

    # Rename cols to better handling
    df = df.rename(
        mapper=rename_rule, 
        axis=1, 
        inplace=False, 
        errors='raise'
    )

    # Reset index values
    df.index = df.index.map(lambda x: x + 1)
    
    # Modify some string value to NaN
    df['genre'] = df['genre'].replace('(no genres listed)', np.nan)

    return df

def check_credit(df):
    ## Add individual credit
    # Count non-na values for each row and compute its proportion
    complete_proportion = df.iloc[:, 1:].count(axis=1) / (df.columns.shape[0] - 1)
    # Apply to df
    df = df.assign(
        **{'ec': np.where(complete_proportion >= 0.5, 5, 0)}
    )

    # Add class-wide credit
    # Count non-na values for each column and compute its proportion
    cross_complete_proportion = df[['color', 'animal', 'movie']].count(axis=0) / df.shape[0]
    # Create binned values for grade threshold
    outs = (
        pd.cut(
            cross_complete_proportion,
            [0, 0.9, 1.01],
            right=False,
            labels=[0, 1]
        )
        .to_numpy()
    )
    # Add ec twice max
    extra_credits = np.clip(np.sum(outs), 0, 2)
    # Apply to df
    df['ec'] += extra_credits

    # Display the result of "cross_complete_proportion"
    print(cross_complete_proportion)

    # Return values
    return df[['name', 'ec']] 


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    # Inner joining op
    inner_joined_df = pd.merge(
        left=pets, 
        right=procedure_history, 
        how='inner', 
        on='PetID'
    )

    # Groupby aggregation op
    return (
        inner_joined_df
        .groupby(by='ProcedureType')['ProcedureType']
        .agg(['count'])['count']
        .sort_values(ascending=False)
        .index[0]
    )

def pet_name_by_owner(owners, pets):
    # Inner join op
    join_df = (
        pd
        .merge(
            left=owners,
            right=pets, 
            how='inner', 
            on='OwnerID'
        )
        .rename(columns={
            'Name_x': 'OwnerName', 
            'Name_y': 'PetName'
        })
    )
    
    # Groupby aggregate op & reset multi-index level
    return (
        join_df
        .groupby(by=['OwnerName', 'Surname'])['PetName']
        .aggregate(lambda x: x.item() if len(x) == 1 else list(x))
        .reset_index(level=1, drop=True)
    )

def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    # Reduce cols to join
    owners = owners[['OwnerID', 'City']]
    pets = pets[['PetID', 'OwnerID']]
    procedure_history = procedure_history[['PetID', 'ProcedureType', 'ProcedureSubCode']]
    procedure_detail = procedure_detail[['ProcedureType', 'ProcedureSubCode', 'Price']]    

    # Join ops
    op_joined = pd.merge(
        left=owners, 
        right=pets, 
        how='inner', 
        on='OwnerID'
    )
    oph_joined = pd.merge(
        left=op_joined,
        right=procedure_history,
        how='left',
        on='PetID'
    )
    ophd_joined = pd.merge(
        left=oph_joined, 
        right=procedure_detail, 
        how='inner',
        on=['ProcedureType', 'ProcedureSubCode']
    )

    # Sort & groupby + aggregate
    return (
        ophd_joined
        .sort_values(
            by=['OwnerID', 'PetID', 'ProcedureType'], 
            ascending=True, 
            na_position='first'
        )
        .groupby(by='City')['Price']
        .sum()
    )


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    return (
        sales
        .groupby(by='Name')['Total']
        .mean(skipna=True)
        .to_frame(name='Average Sales')
    )

def product_name(sales):
    return pd.pivot_table(
        data=sales,
        values='Total',
        index='Name',
        columns='Product',
        aggfunc='sum'
    )

def count_product(sales):
    ...

def total_by_month(sales):
    pass