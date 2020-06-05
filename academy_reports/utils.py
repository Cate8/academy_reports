import os
import numpy as np

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
            stringvalue = stringvalue.replace('[', '')
            stringvalue = stringvalue.replace(']', '')
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