"""
Imports and helpful functions that we use in lectures. Use `make
setup-lec` to copy this to the lecture folders.

Usage:

from course_utils import *
"""

__all__ = [
    "Path",
    "pd",
    "np",
    "plt",
    "sns",
    "display",
    "IFrame",
    "HTML",
    "plotly",
    "px",
    "pio",
    "display_df",
    "dfs_side_by_side",
]

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from IPython.display import HTML, IFrame, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

# Course preferred styles
pio.templates["dsc"] = go.layout.Template(
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
pio.templates.default = "simple_white+dsc"

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
# warnings.simplefilter(action="ignore", category=SyntaxWarning)
# warnings.simplefilter(action="ignore", category=FutureWarning)

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
        {"".join(df.to_html() for df in dfs)}
        </div>
    """
        )
    )
