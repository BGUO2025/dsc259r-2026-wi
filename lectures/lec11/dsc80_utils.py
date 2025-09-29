"""
Imports and helpful functions that we use in DSC 80 lectures. Use `make
setup-lec` to copy this (and custom-rise-styles.css) to the lecture folders.

Usage:

from dsc80_utils import *
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from IPython.display import HTML, IFrame, display
from matplotlib_inline.backend_inline import set_matplotlib_formats
from plotly.subplots import make_subplots

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
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
pio.templates.default = "simple_white+dsc80"

set_matplotlib_formats("svg")
sns.set_context("poster")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# display options for numpy and pandas
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.set_option("display.max_rows", 7)
pd.set_option("display.max_columns", 8)
pd.set_option("display.precision", 2)

# Suppress pandas_tutor syntax warnings
warnings.simplefilter(action="ignore", category=SyntaxWarning)

# Use plotly as default plotting engine
pd.options.plotting.backend = "plotly"


def display_df(
    df, rows=pd.options.display.max_rows, cols=pd.options.display.max_columns
):
    """Displays n rows and cols from df"""
    with pd.option_context(
        "display.max_rows", rows, "display.max_columns", cols
    ):
        display(df)


def dfs_side_by_side(*dfs):
    """
    Displays two or more dataframes side by side.
    """
    display(
        HTML(
            f"""
        <div style="display: flex; gap: 1rem;">
        {''.join(df.to_html() for df in dfs)}
        </div>
    """
        )
    )
