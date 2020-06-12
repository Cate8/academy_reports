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
            except:  # is empty string we need [np.nan]
                val = np.array([np.nan])
        else:
            val = np.array([stringvalue])
        return val

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
                         'A11': '31.20', 'A12': '27.72', 'MA1': '31.2', 'MA2': '37.5', 'MA4': '39.6'}
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


