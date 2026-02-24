# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    # My original choice
    # return ['NMAR', 'MD', 'MAR', 'MCAR', 'MAR']

    # My reasoning to the 4th problem, I don't think series number will correlate to the product itself, 
    # If product is missing, I cannot come up with reasoning it will be missing due to some reason on product itself.
    # But then I realize series number is a column, and product name is not a column, and missing column is the satifaction
    # I think user could be satisfy/dissatisfy based on the product, which is represented by series number

    return ['NMAR', 'MD', 'MAR', 'MAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    # My original choice
    # return ['MCAR', 'MAR', 'NMAR', 'NMAR', 'MD']

    # My reasoning to 1st problem is that I cannot think of how different values from other columns will directly
    # cause address columne to have NaN, so it's not MD; Address value itself cannot cause it to be NaN unless it contains
    # invalid character set, but it will not address otherwise, it's not NMAR; I really cannot think of any relationship
    # from other column to relate to missingness of address; so I chose MCAR at the end ????

    # My reasoning to 3rd problem is that I missed that there is a column  called "number of sport", 
    # if number of sport a student has played is 0, then sport previously played will not be a valid value

    # My reasoning to 5th problem is that I understood as that the student who are UCSD are automatically assigned an
    # unique code, but if the student has not sign up for DUO, then they would not have filled in their phone number. So 
    # Well, I mean this situation is MD <-> there is another observed column that explain whether a student has signed up or not;
    # Well, assuming this is NMAR, the value of the phone will not be causing any missingness of the phone number
    # Yep, I think only choice is MCAR for this case. ???? But the problem is ambiguous

    return ['MAR', 'MAR', 'MD', 'NMAR', 'MCAR']

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def first_round():
    # Get data
    payments_fp = Path('data') / 'payment.csv'
    payments = pd.read_csv(payments_fp)
    payments_test = payments.copy()

    # Preprocess age
    payments_test['age'] = (
        pd.to_datetime(
            payments['date_of_birth'], 
            errors='raise', 
            dayfirst=True, 
            yearfirst=False, 
            format='%d-%b-%Y'
        )
        .dt.year
        .pipe(lambda x: 2024 - x)
    )
    # Preprocessing missingness or not
    payments_test['has_credit_card'] = np.where(payments_test['credit_card_number'].notna(), True, False)

    # # Display distribution
    # create_kde_plotly(
    #     df=payments_test[payments_test['age'].notna()], 
    #     group_col='has_credit_card', 
    #     group1=True, 
    #     group2=False,
    #     vals_col='age'
    # )

    ## Statistical Testing
    # Compute observed test statistics, using groupby method
    observed_group = payments_test.groupby(by='has_credit_card')['age'].mean()
    observed_ts = np.abs(observed_group[True] - observed_group[False])

    # Compute simulated test statistics using permutation test
    n_repetitions = 500
    shuffled = payments_test.copy()
    simulated_ts = []
    for _ in range(n_repetitions):
        # Permutate
        shuffled['has_credit_card'] = np.random.permutation(shuffled['has_credit_card'])
        # Calculate each test statistics, using tedious indexing
        abs_mean = np.abs(
            shuffled[shuffled['has_credit_card']]['age'].mean() - 
            shuffled[~shuffled['has_credit_card']]['age'].mean()
        )
        simulated_ts.append(abs_mean)

    # P-value and conclusion
    p_val = float(np.mean(simulated_ts >= observed_ts))
    sig_level = 0.05
    verdict = 'R' if p_val <= sig_level else 'NR'
    return [p_val, verdict]

    # This is not what I am expected, but the distribution does have very similar mean
    # I had problem where it returns np.true_ instead of true

def second_round():
    # Get data
    payments_fp = Path('data') / 'payment.csv'
    payments = pd.read_csv(payments_fp)
    payments_test = payments.copy()

    # Preprocess age
    payments_test['age'] = (
        pd.to_datetime(
            payments['date_of_birth'], 
            errors='raise', 
            dayfirst=True, 
            yearfirst=False, 
            format='%d-%b-%Y'
        )
        .dt.year
        .pipe(lambda x: 2024 - x)
    )
    # Preprocessing missingness or not
    payments_test['has_credit_card'] = np.where(payments_test['credit_card_number'].notna(), True, False)
    payments_test = payments_test[payments_test['age'].notna()]

    ## Statistical Testing
    # Compute observed test statistics, using groupby method
    observed_group = payments_test.groupby(by='has_credit_card')['age'].mean()
    observed_ts = np.abs(observed_group[True] - observed_group[False])

    # Compute simulated test statistics using permutation test & KS
    simluated_group = payments_test.groupby('has_credit_card')['age']
    simulated_ts = stats.ks_2samp(
            simluated_group.get_group(True), 
            simluated_group.get_group(False)
    ).pvalue

    # P-value and conclusion
    p_val = float(np.mean(simulated_ts >= observed_ts))
    sig_level = 0.05
    verdict = 'R' if p_val <= sig_level else 'NR'
    decision = 'D' if verdict == 'R' else 'ND'
    return [p_val, verdict, decision]


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):
    p_vals = {}
    for col in heights.columns:
        # Ignore irrelevant cols
        if col in ['child', 'father']: continue

        # Get missing-ness property
        heights[col+'_missing'] = np.where(heights[col].isna(), True, False)
        groups = heights.groupby(by=col+'_missing')['father']

        # Test statistics & P-val
        p_val = stats.ks_2samp(
            data1=groups.get_group(True), 
            data2=groups.get_group(False)
        ).pvalue
        p_vals[col] = p_val

    return pd.Series(
        data=p_vals.values(), 
        index=p_vals.keys()
    )


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    new_heights_test = new_heights.copy()

    # Create new feature by using quartile bin
    quartiles = pd.qcut(new_heights_test['father'], q=4)

    # Aggregate to get conditional mean
    groups = new_heights_test.groupby(quartiles)['child'].transform('mean')

    # Statistical imputation
    new_heights_test['child'] = new_heights_test['child'].fillna(groups)

    return new_heights_test['child']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    # Find row with valid values
    valid_child = child.loc[child.notna()]

    # Create probability map
    hist = list(np.histogram(valid_child, 10))
    hist[0] = hist[0] / np.sum(hist[0])
    # Add category/index for the bin, for later sampling purpose
    hist.append(np.array([i for i in range(len(hist[0]))]))

    # Weighted sampling for bin
    bin_sample = np.random.choice(a=hist[2], size=(N,), p=hist[0])
    # Uniform sampling for within-bin value
    value_sample = np.array([])
    for bin_idx in bin_sample:
        lower_bound, higher_bound = hist[1][bin_idx], hist[1][bin_idx+1]
        value = (higher_bound - lower_bound) * np.random.random_sample() + lower_bound
        value_sample = np.append(value_sample, value)
    return value_sample

def impute_height_quant(child):
    # Probabilistic imputation
    imputed_child = quantitative_distribution(child, child.shape[0])
    return child.fillna(pd.Series(imputed_child))


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    return (
        [1, 2, 2, 1],
        ["https://www.fbi.gov/robots.txt", "https://jsonplaceholder.typicode.com/robots.txt"]
    )
