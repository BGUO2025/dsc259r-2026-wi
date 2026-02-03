# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 259R preferred styles
pio.templates["dsc259R"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc259R"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    loans = loans.copy()

    # Prompt 1: Convert to timestamp type
    format = '%b-%Y'
    issue_d_dateTime_col = pd.to_datetime(
        arg=loans['issue_d'],
        errors='raise',
        dayfirst=False,
        yearfirst=False,
        utc=False,
        format=format,
        exact=True,
    )
    loans['issue_d'] = issue_d_dateTime_col

    # Prompt 2: Convert str to int
    loans['term'] = (
        loans['term']
        .str.strip()
        .str.split(' ')
        .transform(lambda x: x[0])
        .astype(np.int32)
    )
    
    # Prompt 3: Clean uncleaned "emp_title"
    loans['emp_title'] = (
        loans['emp_title']
        .str.lower()
        .str.strip()
    )
    rn_mask = loans['emp_title'].str.startswith('rn') & loans['emp_title'].str.endswith('rn')
    loans.loc[rn_mask, 'emp_title'] = 'registered nurse'

    # Prompt 4: Create term end
    loans = loans.assign(**{'term_end': loans.apply(lambda x: x['issue_d'] + pd.DateOffset(months=x['term']) , axis=1)})

    return loans


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def correlations(df, pairs):
    # Separate pairs using zip()
    # Lazy evaluation and it does not execute right away
    col_left, col_right = zip(*pairs)

    # Switch to list to materialize data, so that you can index multi-cols in df
    # It use C-level iteration instead of Python loop
    df_left = df[list(col_left)]
    df_right = df[list(col_right)]

    # Compute correlation
    # Require df_right to have the same col names as df_left
    corr_ser = df_left.corrwith(df_right.set_axis(col_left, axis=1))

    # Rename it
    # Iterate using Python loop, but only on column names
    corr_ser.index = [f"r_{left}_{right}" for left, right in pairs]

    return corr_ser


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    import plotly.express as px

    ## Preprocess df for preparing box plot
    # Convert term to str, to be parsed as categories
    loans['term'] = loans['term'].astype(str)
    # Bin discrete numeric values to categories
    fico_score_status = pd.cut(
        x=loans['fico_range_low'],
        bins=[580, 670, 740, 800, 850],
        labels=['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)'],
        right=False
    )
    loans = loans.assign(**{ 'fico_score_status': fico_score_status})

    ## Box plot
    fig = px.box(
        # Feed data
        loans,
        x='fico_score_status',
        y='int_rate',

        # Set up categories
        category_orders={
            'fico_score_status': ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)'],
            'term ': ['36', '60']
        },

        # Set labels info
        labels = {
            'fico_score_status': 'Credit Score Range',
            'int_rate': 'Interest Rate (%)',
            'term': 'Loan Length (Months)'
        },
        title='Interest Rate Vs. Credit Score',
        
       
        # Set aesthetic styling
        color='term',
        color_discrete_map={
            '36': '#636EFA', 
            '60': '#EF553B'
        },

        # Set size styling
        width=700,
        height=400,
    )
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    has_ps = loans["desc"].notna()
   
    rates = loans["int_rate"].values
    has_ps_array = has_ps.values
   
    observed = rates[has_ps_array].mean() - rates[~has_ps_array].mean()
   
    stats = []

    for _ in range(N):
        shuffled = np.random.permutation(has_ps_array)

        perm_with = rates[shuffled].mean()
        perm_without = rates[~shuffled].mean()
   
        stats.append(perm_with - perm_without)
   
    return float(np.mean(np.array(stats) >= observed))
    
def missingness_mechanism():
    # Assume the p-value is low enough to reject MCAR
    return 2
    
def argument_for_nmar():
    '''
    Put your justification here in this multi-line string.
    Make sure to return your string!
    '''
    return "Applicants might intentionally omit a personal statement if the content of that statement would reveal high financial risk not captured by their numerical data."


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    tax = 0

    for i in range(len(brackets)):
        rate, lower = brackets[i]

        # Upper bound is the next bracket's lower limit
        if i + 1 < len(brackets):
            upper = brackets[i + 1][1]
        else:
            upper = income
       
        taxable = min(income, upper) - lower # Amount taxed in this bracket

        if taxable > 0:
            tax += taxable * rate
   
    return tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    df = state_taxes_raw.copy()

    # Eliminate all the empty rows
    df = df.dropna(how="all")

    # Eliminate "junk"
    is_junk = df["State"].astype(str).str.contains(r"\(", na=False)
    df.loc[is_junk, "State"] = pd.NA
   
    # Fill state names
    df["State"] = df["State"].ffill()

    # Clean the Rate column strings
    df["Rate"] = (
        df["Rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
   
    # Convert to numeric
    df["Rate"] = pd.to_numeric(df["Rate"], errors='coerce').fillna(0)
   
    # Drop rows that do not have a Rate
    df = df.dropna(subset=["Rate"])
   
    # Convert Rate to proportion float
    df["Rate"] = df["Rate"].div(100).round(2)
   
    # Convert the Lower Limit
    df["Lower Limit"] = (
        pd.to_numeric(df["Lower Limit"], errors='coerce')
        .fillna(0)
        .astype(int)
    )

    df = df[["State", "Rate", "Lower Limit"]]
   
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    return(
        state_taxes
        .assign(tup=lambda x:list(zip(x["Rate"], x["Lower Limit"])))
        .groupby("State")["tup"]
        .apply(list)
        .to_frame("bracket_list")
    )
    
def combine_loans_and_state_taxes(loans, state_taxes):        
    # Now it's your turn:
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
       
    # Perform invert mapping from full to abbreviation
    inv_map = {v: k for k, v in state_mapping.items()}

    st_clean = state_taxes.copy()
    st_clean["State"] = st_clean["State"].map(inv_map)

    brackets = state_brackets(st_clean) # Build Brackets

    out = loans.copy()
    out = out.rename(columns={"addr_state": "State"})
   
    merged = out.merge(brackets, on="State", how="left") # Merge

    return merged


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]

    def compute_tax(income, brackets):
        if not isinstance(brackets, list):
            return 0
       
        tax = 0
       
        for i, (rate, lower) in enumerate(brackets):
            if i + 1 < len(brackets):
                upper = brackets[i+1][1]
            else:
                upper = income
           
            taxable = max(0, min(income, upper) - lower)
            tax += taxable * rate

        return tax
   
    out = loans_with_state_taxes.copy()

    out["federal_tax_owed"] = out["annual_inc"].apply(
        lambda x: compute_tax(x, FEDERAL_BRACKETS)
    )
   
    out["state_tax_owed"] = out.apply(
        lambda row: compute_tax(row["annual_inc"], row["bracket_list"]),
        axis=1
    )
   
    out["disposable_income"] = (
        out["annual_inc"] - out["federal_tax_owed"] - out["state_tax_owed"]
    )
   
    return out


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    result = None

    for kw in keywords:
        # Filter by keyword in the job title
        mask = loans["emp_title"].str.contains(kw, na=False)
        filtered = loans[mask]

        # Group by category and find the mean
        grouped = filtered.groupby(categorical_column)[quantitative_column].mean()

        # Find the overall mean
        overall = filtered[quantitative_column].mean()
        grouped.loc["Overall"] = overall

        col_name = f"{kw}_mean_{quantitative_column}" # Name column
        col_loans = grouped.to_frame(name=col_name) # Convert to DataFrame

        # Join columns together
        if result is None:
            result = col_loans
        else:
            result = result.join(col_loans)
   
    return result.round(2)


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def simple_aggregate(loans, keywords, quantitative_column, categorical_column):
    result = None
   
    for kw in keywords:
        f = loans[loans["emp_title"].str.contains(kw, na=False)]
        g = f.groupby(categorical_column)[quantitative_column].mean()
        g.loc["Overall"] = f[quantitative_column].mean()
        name = f"{kw}_mean_{quantitative_column}"
        g = g.to_frame(name)
        result = g if result is None else result.join(g)
   
    return result

def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    agg = simple_aggregate(loans, keywords, quantitative_column, categorical_column)
   
    A = agg.iloc[:-1, 0]
    B = agg.iloc[:-1, 1]
   
    overall_A = agg.iloc[-1, 0]
    overall_B = agg.iloc[-1, 1]
   
    result = ((A > B).all() and overall_A < overall_B) or \
             ((A < B).all() and overall_A > overall_B)
   
    return bool(result)
    
def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': ['engineer', 'nurse'],
        'quantitative_column': 'loan_amnt',
        'categorical_column': 'home_ownership'
    }