import os
import numpy as np
import pandas as pd


# GET LIST OF PATHS SORTED BY NAME
def path_generator(path, pattern):
    paths = []
    for root, _, file in os.walk(path):
        for f in file:
            if f.endswith(pattern):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# CONVERT STRING COLUMNS OF DF TO LISTS
def convert_strings_to_lists(df, columns):
    """
    If the csv contains a column that is ',' separated, that column is read as a string.
    We want to convert that string to a list of values. We try to make the list float or string.
    """

    def tolist(stringvalue):
        if isinstance(stringvalue, str):
            try:
                stringvalue = stringvalue.split(sep=',')
                try:
                    val = np.array(stringvalue, dtype=float)
                except:
                    val = np.array(stringvalue)
            except:
                val = np.array([])
        elif np.isnan(stringvalue):
            return np.array([])
        else:
            val = np.array([stringvalue])
        return val.tolist()

    for column in columns:
        df[column] = df[column].apply(tolist)
    return df


# UNNESTING LISTS IN COLUMNS DATAFRAMES
def unnesting(df, explode):
    """
    Unnest columns that contain list creating a new row for each element in the list.
    The number of elements must be the same for all the columns, row by row.
    """
    length = df[explode[0]].str.len()
    idx = df.index.repeat(length)
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    finaldf = df1.join(df.drop(explode, 1), how='left')
    finaldf.reset_index(drop=True, inplace=True)

    length2 = [list(range(l)) for l in length]
    length2 = [item + 1 for sublist in length2 for item in sublist]
    name = explode[0] + '_index'
    finaldf[name] = length2

    for column in finaldf.columns:
        try:
            if set(finaldf[column]) <= {'True', 'False', 'nan'}:
                replacing = {'True': True, 'False': False, 'nan': np.nan}
                finaldf[column] = finaldf[column].map(replacing)
        except:
            pass
    return finaldf

# BASAL WEIGHTS
def relative_weights(subject, weight):
    basal_weights = {'A5': '32.68', 'A6': '31.46', 'A7': '30.40', 'A8': '31.38', 'A9': '31.65', 'A10': '27.71',
                         'A11': '31.20', 'A12': '27.72', 'MA1': '25.33', 'MA2': '21.97', 'MA3': '24.93', 'MA4': '23.31', 'MA5': '25.28'}
    for key, value in basal_weights.items():
        if subject == key:
            basal_weight_subj = float(value)
            relative_weight_subj = weight / basal_weight_subj * 100
            return relative_weight_subj

# COMPUTE WINDOW AVERAGE
def compute_window(data, runningwindow):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    for i in range(len(data)):
        if i < runningwindow:
            performance.append(round(np.mean(data[0:i + 1]), 2))
        else:
            performance.append(round(np.mean(data[i - runningwindow:i]), 2))
    return performance

# COLLECT ALL REPONSES TIMES IN A COLUMN
def create_responses_time(row):
    try:
        result = row['STATE_Incorrect_START'].tolist().copy()
    except (TypeError, AttributeError):
        result = row['STATE_Incorrect_START'].copy()

    items = [row['STATE_Correct_first_START'], row['STATE_Correct_other_START'], row['STATE_Punish_START']]

    for item in items:
        if not np.isnan(item):
            result += [item]

    return result

# RESPONSE RESULT COLUMN
def create_reponse_result(row):
    result = ['incorrect'] * len(row['STATE_Incorrect_START'])
    if row['trial_result'] != 'miss' and row['trial_result'] != 'incorrect':
        result += [row['trial_result']]
    return result

# CREATE CSVS
def create_csv(df, path):
    df.to_csv(path, sep=';', na_rep='nan', index=False)

# CHANCE CALCULATION
def chance_calculation(correct_th):
    screen_size = 1440 * 0.28
    chance = 1 / (screen_size / correct_th)
    return chance

# PECRCENTAGE AXES
def axes_pcent(axes, label_kwargs):
    """
    convert y axis form 0-1 to 0-100%
    """
    # convert y axis form 0-1 to 0-100%
    axes.set_ylabel('Accuracy (%)', label_kwargs)
    axes.set_ylim(0, 1.1)
    axes.set_yticks(np.arange(0, 1.1, 0.1))
    axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])


