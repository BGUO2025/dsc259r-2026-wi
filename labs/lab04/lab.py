# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def helper_convert_datetime(df, col):
    # Conver to datetime object using format
    format = '%Y-%m-%d %H:%M:%S'
    time_col = (
        pd.to_datetime(
            df[col], 
            errors='raise',
            dayfirst=False,
            yearfirst=True, 
            format=format,
            exact=True
        )
    )
    # Transform its column to datetime type
    return df.assign(**{col: time_col})

def helper_filter_prime_time(df, start, end):
    # Define filtering mask
    mask = (
        df['Time']
        .dt.hour
        .between(start, end, inclusive='left')
    )
    
    # Start marking all time without prime time as nan
    df.loc[~mask, 'Time'] = np.nan
    return df

def prime_time_logins(login):
    return (
        login
        # Convert to datetime object
        .pipe(helper_convert_datetime, 'Time')
        # Mark filtered row as nan
        .pipe(helper_filter_prime_time, 16, 20)
        # Convert further to datetime date
        .assign(**{'Time': lambda df: df['Time'].dt.date})
        # Groupby, while keeping Login Id with one nan row
        .groupby(by='Login Id', dropna=False)
        # Aggregate, Ignore nan as count of row
        .count()
    )


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    # Set up today time in datetime
    from datetime import datetime
    today_time = datetime(2024, 1, 31, hour=11, minute=59, second=0)

    return (
        login
        # Convert to datetime object
        .pipe(helper_convert_datetime, 'Time')
        # Add new col that find the duration of first login to current
        # Conver to day with truncation
        .assign(**{'Time_Diff': lambda df: (today_time - df['Time']).dt.days})
        # Groupby
        .groupby(by='Login Id')['Time_Diff']
        # Aggregte, numbers of logins per day, duration + 1 because they are still member today
        .aggregate(lambda ser: ser.shape[0] / (ser.max() + 1))
    )


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]
                         
def cookies_p_value(N):
    # H0: p(burnt) 4%
    # H1: p(burnt) >= 4%
    # Observation: Several students report dissatisfaction
    # Test statistics: Num Burned Cookies
    # Observed Distribution: Like (250-15) / 250; Dislike 15/ 250
    # Null Distribution: Like 0.96; Dislike 0.04

    # Null and Observed Distributions
    null_dist = np.array([0.96, 0.04])
    obs_ts = 15
    obs_sample_sz = 250

    # Simulate N trials of Bernoulli Distribution
    simulated_samples = np.random.binomial(obs_sample_sz, null_dist[1], N)

    # Compute N test statistics/empirical distribution of test statistics
    emp_ts = simulated_samples

    # Compute P-value
    return np.mean(emp_ts >= obs_ts)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 3, 5, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 3


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1, 2]
    
def bhbe_col(heroes):
    return (
        heroes['Eye color']
        .str.contains('blue', case=False) & 
        heroes['Hair color']
        .str.contains('blond', case=False)
    )

def superheroes_observed_statistic(heroes):
    bhbe_heroes = heroes[bhbe_col(heroes)]
    return np.mean(bhbe_heroes['Alignment'] == 'good')

def simulate_bhbe_null(heroes, N):
    # Numbers of heros with blonde and blue eyes
    mask_len = bhbe_col(heroes).sum()
    good_heroes = heroes['Alignment'] == 'good'

    # Generate samples from null distribution
    # You can use np.random.binomial(), it's actually more effective
    # I want to have diverse problem-solving here
    simulated_samples = np.random.choice(good_heroes, size=(N, mask_len), replace=True)
    return simulated_samples.mean(axis=1)

def superheroes_p_value(heroes):
    # Compute observed test statistics
    obs_ts = superheroes_observed_statistic(heroes)
    # Compute empirical test statistics
    simulated_ts = simulate_bhbe_null(heroes, 100000)
    # Compute p-value
    p_val = np.mean(simulated_ts >= obs_ts)

    # Define significance level
    sig_level = 0.01
    # Verdict
    verdict = 'Reject' if p_val <= sig_level else 'Fail to reject'
    return (p_val, verdict)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    # Compute test statistics
    return (
        data
        .groupby(by='Factory')[col]
        .mean()
        .aggregate(lambda x: abs(x.loc['Waco'] - x.loc['Yorkville']))
    )

def simulate_null(data, col='orange'):
    shuffle_data = data.copy()

    # Compute shuffled sample from permutation test 
    shuffle_data['Factory'] = np.random.permutation(shuffle_data['Factory'])

    # Compute empirical distribution from test statistics
    return diff_of_means(shuffle_data, col)

def color_p_value(data, col='orange'):
    # Compute observed test statistics
    obs_ts = diff_of_means(data, col)
    # Compute empirical test statistics
    simulated_ts = [simulate_null(data, col) for _ in range(1000)]
    # Compute p-values
    return np.mean(simulated_ts >= obs_ts)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [
        ('yellow', 0.0),
        ('orange', 0.047),
        ('red', 0.245),
        ('green', 0.45),
        ('purple', 0.979)
    ]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------

    
def same_color_distribution():
    return (0.007, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
