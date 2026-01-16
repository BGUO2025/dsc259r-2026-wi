# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    import re

    # Initialize names
    key_names = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    base_names = ['Midterm', 'Final', 'lab', 'project', 'discussion', 'checkpoint']
    all_names = grades.columns.to_list()
    valid_names = []

    # Specify RE patterns
    pattern_str1 = '(' + '|'.join(base_names[:2]) + ')$'
    pattern_str2 = r'(' + '|'.join(base_names[2:]) + r')\d{2}$'

    # Find valid names
    matched = [re.findall(pattern=pattern_str1 + '|' + pattern_str2, string=name) for name in all_names]
    for i, match_arr in enumerate(matched):
        if not match_arr: continue
        valid_names.append(all_names[i])

    # Assign them to dict
    syllabus_names = {}
    for key_name in key_names:
        candidates = []
        for candidate in valid_names:
            if key_name == 'project':
                if candidate.lower().startswith('project') and \
                candidate.lower().find('checkpoint') == -1:
                    candidates.append(candidate)
            elif candidate.lower().find(key_name) != -1:
                candidates.append(candidate)

        syllabus_names[key_name] = candidates

    return syllabus_names


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    # Get useful array
    course_names = get_assignment_names(grades)
    col_names = set(grades.columns)
    grades_test = grades.set_index(keys='PID')

    # Initialize dict to store person-project score
    project_grades_per_student = {}

    # Iterate student by ID
    for id in grades_test.index:
        projects = np.array([])
        # Reset individual Project score
        auto_pt, auto_max, free_pt, free_max = 0, 0, 0, 0
        for project in course_names['project']:
            # Collect individual score while handling NA value and field not found
            auto_pt = 0 if pd.isna(grades_test.loc[id, project]) \
                else grades_test.loc[id, project]
            auto_max = 0 if pd.isna(grades_test.loc[id, project+' - Max Points']) \
                else grades_test.loc[id, project+' - Max Points']
            free_pt = 0 if not project+'_free_response' in col_names \
                else (0 if pd.isna(grades_test.loc[id, project+'_free_response'])
                    else grades_test.loc[id, project+'_free_response'])
            free_max = 0 if not project+'_free_response - Max Points' in col_names \
                else (0 if pd.isna(grades_test.loc[id, project+'_free_response - Max Points']) \
                    else grades_test.loc[id, project+'_free_response - Max Points'])

            # Compute individual project raw score
            raw_score = (auto_pt + free_pt) / (auto_max + free_max)

            # Collect raw score
            projects = np.append(projects, raw_score)

        # Compute all project score for each student 
        project_grades_per_student[id] = projects.mean()

    return pd.Series(project_grades_per_student.values())


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    # Convert them to time delta type to calculate time diff
    Lateness_dt = (
        pd.to_timedelta(col)
        .dt.total_seconds()
    )

    # cut() converts the values into bin
    Lateness_rate = pd.cut(
        x=Lateness_dt,
        bins=[0, 26*3600, 24*7*3600, 24*14*3600, np.inf],
        right=False,
        labels=[1, 0.9, 0.7, 0.4],
        duplicates='raise'
    )

    return Lateness_rate


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    # Get useful array
    course_names = get_assignment_names(grades)
    binary_mask = grades.columns.str.contains('lab')
    lab_grades = grades.loc[:, binary_mask].copy()

    # Iterate lab assignemnts
    for lab in course_names['lab']:
        # Fill NA values
        lab_grades.fillna({lab: 0}, inplace=True)

        # Transform lab col by multiplying original value by lateness penalty rate
        lab_grades = lab_grades.assign(
            **{lab: lab_grades[lab] * lateness_penalty(lab_grades[lab + ' - Lateness (H:M:S)']).astype(np.float64)}
        )

        # Normalize the score between 0 - 1
        max_score = lab_grades[lab].max()
        lab_grades = lab_grades.assign(
            **{lab: lab_grades[lab] / max_score}
        )

    return lab_grades[course_names['lab']]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    # Get useful array
    course_names = get_assignment_names(processed)

    # Gather each lab score into a list
    lab_scores_collection = processed.apply(
        func=lambda row: [row[lab] for lab in course_names['lab']],
        axis=1
    )

    # Drop the worse score and take the average
    def drop_last_and_compute_mean(row):
        row.sort(reverse=False)
        row.pop(0)
        return sum(row) / len(row)
    lab_scores_mean = lab_scores_collection.apply(
        func=drop_last_and_compute_mean,
    )

    return lab_scores_mean


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def process_scores(grades, course_names, course_cat):
    # Get useful array
    binary_mask = grades.columns.str.contains(course_cat)
    course_grades = grades.loc[:, binary_mask].copy()

    # Iterate courses
    for course in course_names:
        pt_col, max_col = course, course +  ' - Max Points'

        # Fill NA values
        course_grades.fillna({pt_col: 0}, inplace=True)
        course_grades.fillna({max_col: 0}, inplace=True)

        # Compute the score: original / max and make a new col
        course_grades = course_grades.assign(
            **{course + ' - Final Points': np.where(
                course_grades[max_col] != 0, 
                course_grades[pt_col] / course_grades[max_col],
                0
            )}
        )
    
    # Filter out irrelevant cols
    binary_mask_final = course_grades.columns.str.endswith(' - Final Points')
    return course_grades.loc[:, binary_mask_final]

def total_scores(processed, course_names, course_cat):
    # Gather each course cat score into a list
    course_scores_collection = processed.apply(
        func=lambda row: [row[course+' - Final Points'] for course in course_names],
        axis=1
    )

    # Take the average across courses
    course_scores_mean = course_scores_collection.apply(
        func=lambda row: sum(row) / len(row),
    )

    return course_scores_mean

def total_except_midterm(grades):
    # Get course names
    course_names = get_assignment_names(grades)

    # Score for projects
    mean_project = projects_total(grades)

    # Score for labs
    processed_lab = process_labs(grades)
    mean_lab = lab_total(processed_lab)

    # Score for checkpoints
    processed_cp = process_scores(grades, course_names['checkpoint'], 'checkpoint')
    mean_cp = total_scores(processed_cp, course_names['checkpoint'], 'checkpoint')

    # Score for discussions
    processed_disc = process_scores(grades, course_names['disc'], 'discussion')
    mean_disc = total_scores(processed_disc, course_names['disc'], 'discussion')

    # Score for final exam
    processed_final = process_scores(grades, course_names['final'], 'Final')
    mean_final = total_scores(processed_final, course_names['final'], 'Final')

    return (course_names, mean_project, mean_lab, mean_cp, mean_disc, mean_final)

def compute_final_scores(mean_lab, mean_project, mean_cp, mean_disc, mean_midterm, mean_final):
    expected_score \
        = 0.2 * mean_lab \
        + 0.3 * mean_project \
        + 0.025 * mean_cp \
        + 0.025 * mean_disc \
        + 0.15 * mean_midterm \
        + 0.3 * mean_final
    return expected_score

def total_points(grades):
    # Score for non-midterm categories
    (course_names, mean_project, mean_lab, mean_cp, mean_disc, mean_final) = total_except_midterm(grades)

    # Score for midterm
    processed_midterm = process_scores(grades, course_names['midterm'], 'Midterm')
    mean_midterm = total_scores(processed_midterm, course_names['midterm'], 'Midterm')

    # Final Score
    return compute_final_scores(mean_lab, mean_project, mean_cp, mean_disc, mean_midterm, mean_final)

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    # Use cut() to separate values into bins
    return pd.cut(
        x=total,
        bins=[-np.inf, 0.6, 0.7, 0.8, 0.9, np.inf],
        right=False,
        labels=['F', 'D', 'C', 'B', 'A'],
        duplicates='raise'
    )

def letter_proportions(total):
    # Get input values
    letter_grades = final_grades(total)
    all_cnt = letter_grades.shape[0]

    # Perfor Split (group-by); Apply(aggregate & transform); Combine(sort)
    return (letter_grades
        .groupby(by=letter_grades)
        .aggregate(func=['count'])
        .transform(func=lambda letter_cnt: letter_cnt / all_cnt)
        .sort_values(by='count', ascending=False)  
    )

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    # Create binary mask for redemption questions
    mask = np.full(
        shape=(final_breakdown.shape[1],),
        fill_value=False
    )
    mask[0] = True
    for question in question_numbers:
        mask = mask | final_breakdown.columns.str.startswith('Question '+ str(question) + ' ')
    
    # Create new df with ONLY redemption question col names
    red_breakdown = final_breakdown.loc[:, mask].copy()
    red_cols = red_breakdown.columns

    # Fill NA values
    red_breakdown.fillna(
        {col : 0 for col in red_cols}, 
        inplace=True
    )

    # Extract max redemption scores
    red_max_score = (
        red_breakdown
        .iloc[:, 1:]
        .max(axis=0)
        .sum()
    )

    # Extract sum redemption scores
    red_scores = (
        red_breakdown
        .iloc[:, 1:]
        .sum(axis=1) / red_max_score
    )

    # Create new cols for raw scores
    red_breakdown = red_breakdown.assign(
        **{'Raw Redemption Score': red_scores}
    )

    # Extract result df
    return red_breakdown[['PID', 'Raw Redemption Score']]
    
def combine_grades(grades, raw_redemption_scores):
    # Convert type to prevent df.join() error, but I am using pd.merge now anyway
    grades['PID'] = grades['PID'].astype(str)
    raw_redemption_scores['PID'] = raw_redemption_scores['PID'].astype(str)

    # FROM grades g INNER JOIN raw_redemption_scores r ON g.PID = r.PID
    return pd.merge(
        left=grades,
        right=raw_redemption_scores,
        how='inner',
        on='PID',
        validate='one_to_one'
    )


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    mean_val = ser.mean()
    std_val = ser.std(ddof=0)
    ser = ser.transform(
        func=lambda val: (val - mean_val) / std_val
    )
    return ser
    
def add_post_redemption(grades_combined):
# I either encounter problem:
#   with NOT 0-1 for value range (using mean/std from midterm score proportion regardless whether raw_redemption overtook) OR
#   pre_redemption > post_redemption (using mean/std from respective field )
# max: np.float64(1.027390041308168)

    ## pre-redemption
    # Fill NA for midterm
    grades_combined.fillna({'Midterm': 0}, inplace=True)

    # Compute score proportions for both
    midterm_score_proportion = grades_combined['Midterm'] / grades_combined['Midterm'].max()
    raw_red_score_proportion = grades_combined['Raw Redemption Score'] / grades_combined['Raw Redemption Score'].max()

    # Add the pre-redemption col
    grades_combined = grades_combined.assign(
        **{'Midterm Score Pre-Redemption': midterm_score_proportion}
    )

    ## post-redemption
    # Compute z-scores for both
    midterm_z_score = z_score(midterm_score_proportion)
    raw_red_z_score = z_score(raw_red_score_proportion)

    # Compute mean/std for both
    midterm_score_mean = midterm_score_proportion.mean()
    midterm_score_std = midterm_score_proportion.std(ddof=0)

    # Redemption policy
    finalized_proportion = np.where(
        raw_red_z_score > midterm_z_score,
        raw_red_z_score * midterm_score_std + midterm_score_mean,
        midterm_z_score * midterm_score_std + midterm_score_mean
    )
    finalized_proportion = np.where(
        finalized_proportion > 1,
        1,
        finalized_proportion
    )

    # Add the post-redemption col
    grades_combined = grades_combined.assign(
        **{'Midterm Score Post-Redemption': finalized_proportion}
    )

    # display(grades_combined['Midterm Score Pre-Redemption'].max())
    # display(grades_combined['Midterm Score Pre-Redemption'].min())

    # display(grades_combined['Midterm Score Post-Redemption'].max())
    # display(grades_combined['Midterm Score Post-Redemption'].min())

    # display(np.all(grades_combined['Midterm Score Pre-Redemption'] <= grades_combined['Midterm Score Post-Redemption']))

    return grades_combined


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    # Gather important info
    grades_combined_red = add_post_redemption(grades_combined)

    # Score for non-midterm categories
    (_, mean_project, mean_lab, mean_cp, mean_disc, mean_final) = total_except_midterm(grades_combined_red)

    # Score for midterm
    mean_midterm = grades_combined_red['Midterm Score Post-Redemption']

    # Final Score with Redemption
    return compute_final_scores(mean_lab, mean_project, mean_cp, mean_disc, mean_midterm, mean_final)
    
def proportion_improved(grades_combined):
    # Extract total points
    original_total_points = total_points(grades_combined)
    redemption_total_points = total_points_post_redemption(grades_combined)

    # Convert it to letter grade
    original_letter_grades = final_grades(original_total_points)
    redemption_letter_grades = final_grades(redemption_total_points)

    # Find the proportion of difference in letter grade
    return np.mean(original_letter_grades != redemption_letter_grades)


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    # Extract total points
    original_total_points = total_points(grades_analysis)
    redemption_total_points = total_points_post_redemption(grades_analysis)

    # Convert it to letter grade
    original_letter_grades = final_grades(original_total_points)
    redemption_letter_grades = final_grades(redemption_total_points)

    # Compute if they are equal or not
    grades_analysis = grades_analysis.assign(
        **{'is_improved': original_letter_grades != redemption_letter_grades}
    )

    # Groupby & aggregate
    return (
        grades_analysis
        .groupby(by='Section')
        ['is_improved']
        .mean()
        .idxmax()
    )
    
def top_sections(grades_analysis, t, n):
    return (
        grades_analysis[grades_analysis['Total Points Post-Redemption'] >= t]
        .groupby(by='Section')
        .size()
        .loc[lambda counts: counts >= n]
        .index
        .to_numpy()
    )


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    return (
        # First groupby + transform (find the rank place and + 1 as it start at 0)
        grades_analysis
        .groupby(by='Section')
        # Speed up performance
        ['Total Points Post-Redemption']
        .transform(lambda x: x.argsort() + 1)
        # Add the transformed col as new col
        .pipe(lambda ser: grades_analysis.assign(** {'Section Rank': ser}))
        # Create pivot table, using new col as index and section as column, aggre by identity()
        .pivot_table(
            index='Section Rank',
            columns='Section',
            values ='PID',
            aggfunc=lambda x: x,
            fill_value=""
        )
    )


# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    # Reasoning on color pattelet, I used viridis as it's unclassed sequential gradient color scale and 
    # friendly to color-blind individual
    pivot_table = (
        grades_analysis
        .pivot_table(
            index='Letter Grade Post-Redemption',
            columns='Section',
            values ='PID',
            aggfunc='count'
        )
        .sort_index(ascending=False)
    )
    pivot_table = pivot_table / pivot_table.sum(axis=0)

    fig = px.imshow(
        pivot_table,
        labels = dict(x="Section", y="Letter Grade Post-Redemption", color="Counts"),
        x = pivot_table.columns,
        y = pivot_table.index,
        color_continuous_scale='Viridis',
        title='Distribution of Letter Grades by Section'
    )

    return fig

