# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np

np.set_printoptions(legacy="1.21")


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    # Use two pointer method to parse consecutive element
    for indx in range(0, len(ints)-1):
        # If previous-current OR current-previous == 1, return True
        if abs(ints[indx] - ints[indx+1]) == 1:
            return True
    # Return false if parsed everything
    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    def compute_mean(nums):
        # Formula for Mean
        return sum(nums) / len(nums)

    def compute_median(nums):
        length = len(nums)
        # Find middle index
        mid = length // 2
        # Formula for median
        if length % 2 != 0:
            return nums[mid]  
        else:
            return compute_mean([nums[mid], nums[mid+1]])
    
    return compute_median(nums) <= compute_mean(nums)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    consec_prefix = ""
    # Find substring with prefix from start to 0, start to 1, ..., start to n
    for curr_n in range(n, 0, -1):
        # reversed the substring and put it together
        consec_prefix += s[0 : curr_n]
    return consec_prefix


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    def expand_by_n(ints, n):
        expanded = []
        for num in ints:
            curr_expand = [expanded_num for expanded_num in range(num - n, num + n + 1)]
            expanded.extend(curr_expand)
        return expanded

    def max_num_digit(num):
        exp = 1
        while num >= (10 ** exp):
            exp += 1
        return exp
    
    def expand_to_str(expanded, exp):
        return list(map(lambda ele: str(ele).rjust(exp, '0'), expanded))
    
    def create_separate_expand_str(strs, n):
        final_expanded = []
        multiple = 1
        start = 0
        end = multiple * (n * 2 + 1)
        while end <= len(strs):
            curr_expand_str = ' '.join(strs[start : end])
            final_expanded.append(curr_expand_str)
            multiple += 1
            start = end
            end = multiple * (n * 2 + 1)
        return final_expanded

    expanded = expand_by_n(ints, n)
    exp = max_num_digit(max(expanded))
    expanded_str = expand_to_str(expanded, exp)
    final_expanded = create_separate_expand_str(expanded_str, n)
    return final_expanded

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    chars = ''
    for line in fh:
        if line[-1] == "\n":
            chars += line[-2]
        else:
            chars += line[-1]
    return chars


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    arr_flat = A.ravel()
    index_arr_flat = np.indices(arr_flat.shape).ravel()
    squared_flat = np.sqrt(index_arr_flat)
    return arr_flat + squared_flat


def where_square(A):
    squared = np.sqrt(A)
    return (squared % 1) == 0


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return matrix
    filtered = list()
    row_size = len(matrix)
    col_size = len(matrix[0])
    for j in range(col_size):
        sum = 0
        for i in range(row_size):
            if not np.isnan(matrix[i][j]): sum += matrix[i][j]
        if sum / row_size <= cutoff:
            continue
        if not filtered: filtered.extend([[] for i in range(row_size)])
        for i in range(row_size):
            filtered[i].append(matrix[i][j])
    return np.array(filtered)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    mean_arr = np.nanmean(matrix, axis=0)
    mean_mask = mean_arr > cutoff
    return matrix[:, mean_mask]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    window_view = np.lib.stride_tricks.sliding_window_view(A, 2)
    growth = (window_view[:, 1] - window_view[:, 0]) / window_view[:, 0]
    return growth.round(2)


def with_leftover(A):
    leftover = 20 % A
    leftover_cumu = np.cumsum(leftover)
    buy_full_share = leftover_cumu > A
    if np.all(buy_full_share == False): return -1
    buy_day = np.min(np.nonzero(buy_full_share))
    return buy_day.astype(int)


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    # Initialize hashmap to store info
    stats_dict = {}

    # Number of players
    stats_dict['num_players'] = salary.shape[0]

    # Number of teams
    stats_dict['num_team'] = salary['Team'].nunique()

    # Total salary
    stats_dict['total_salary'] = salary['Salary'].sum(axis=0)

    # Highest salary
    stats_dict['highest_salary'] = salary['Salary'].max(axis=0)

    # Average salary in team "LA Lakers"
    stats_dict['avg_los'] = (   
        salary
        .groupby(by='Team')
        ['Salary']
        .mean()
        .round(decimals=2)
        .loc['Los Angeles Lakers']
    )

    # Player with fifth lowest salary
    fifth_low_info = salary.sort_values(by='Salary').iloc[4, :]
    stats_dict['fifth_lowest'] = (fifth_low_info['Player'], fifth_low_info['Team'])

    # Duplicate player last name
    name_with_suffices = (
        salary['Player']
        .str.lower()
        .str.endswith((' jr.', ' sr.', ' v', ' iv', ' iii', ' ii', ' i'))
    )
    salary.loc[name_with_suffices, 'Player'] = (
        salary
        .loc[name_with_suffices, 'Player']
        .str.rsplit(pat=' ', n=1, expand=False)
        .str[0]
    )
    are_dupliates = salary.shape[0] == (
        salary['Player']
        .str.split(pat=' ', expand=False)
        .str[-1].nunique()
    )
    stats_dict['duplicates'] = are_dupliates

    # Total salary of the team with highest salary
    stats_dict['total_highest'] = (
        salary[['Team', 'Salary']]
        .groupby(by='Team')
        ['Salary']
        .aggregate(['max', 'sum'])
        .sort_values(by='max', ascending=False)
        .reset_index()
        .loc[0, 'sum']
    )

    return pd.Series(stats_dict)



# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    import re
    from io import StringIO

    # Load the raw string
    fp = Path("data") / "malformed.csv"
    with open(fp, 'r') as f:
        raw_str = f.read()

    # Remove repeating commas
    raw_str = re.sub(pattern=r',{2,}', repl=',', string=raw_str)

    # Remove repeating quotes
    raw_str = re.sub(pattern=r'"{2,}', repl='"', string=raw_str)

    # Remove quotes that appear after alphanumeric/period and before comma
    raw_str = re.sub(pattern=r'(?P<last_char>[\w\.])"+(?P<comma>,)', repl=r'\g<last_char>\g<comma>', string=raw_str)

    # Remove commas that appear at the end of the row
    raw_str = re.sub(pattern=r'(?P<last_num>[-\d+\.\d+]),(?P<new_line>\n)', repl=r'\g<last_num>"\g<new_line>', string=raw_str)

    # Remove commas & quotes that appear at the beginning of the row
    raw_str = re.sub(pattern=r'(?P<new_line>\n),"*(?P<last_char>[\w\.])', repl=r'\g<new_line>\g<last_char>', string=raw_str)
    raw_str = re.sub(pattern=r'(?P<new_line>\n)",*(?P<last_char>[\w\.])', repl=r'\g<new_line>\g<last_char>', string=raw_str)

    # Identify missing quotes and add it
    raw_str = re.sub(pattern=r'(?P<last_num>[-\d+\.\d+])(?P<new_line>\n)', repl=r'\g<last_num>"\g<new_line>\n', string=raw_str)

    # Specify columns types while writing to Pandas df
    df = pd.read_csv(StringIO(raw_str), dtype={
        'first': str,
        'last': str,
        'weight': float,
        'height': float,
        'geo': str
    })

    return df