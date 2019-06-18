'''
Code to Check Date Distribution and Overlap

Katy Koenig
June 2019
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_FILES = ['mod02_original.txt']
VAL_FILES = ['label1.txt']
CLUST_FILES = [x for x in os.listdir() if 'txt' in x and x not in TRAIN_FILES \
               and x not in VAL_FILES]
ALL_FILES = [x for x in os.listdir() if 'txt' in x]


def make_df(file_lst=ALL_FILES):
    '''
    Creates a dataframe of dates from a given list of txt files

    Input:
        file_lst: list of txt files

    Output:
        df: a pandas dateframe with one column
    '''
    for file in file_lst:
        if file == file_lst[0]:
            df = pd.read_fwf(file, header=None)
        else:
            df = pd.concat([df, pd.read_fwf(file, header=None)])
    df.rename(columns={0: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df


def check_date_dist(df, output_filename):
    '''
    Checks the distribution of dates "randomly" generated for uniform
    distribution

    Inputs: a list of dates with each date as a string
            e.g. the output of gen_randomDate

    Outputs:
        Grouped: the groupby object with the count of each date
        A saved png of the plot of distribution of dates by year
    '''
    grouped = df.groupby(df['date'].dt.year).count()
    dist_plt = grouped.plot(kind='bar')
    plt.title("Distribution of Dates")
    dist_plt.figure.savefig(output_filename, bbox_inches="tight")
    plt.clf()
    return grouped


def find_duplicates(list1=TRAIN_FILES, list2=CLUST_FILES+VAL_FILES):
    '''
    Checks for duplicates between two lists

    Inputs:
        list1: list of filenames
        list2: list of filenames

    Output: a set of duplicate values
    '''
    df1 = make_df(list1)
    df2 = make_df(list2)
    return set(df1['date']).intersection(set(df2['date']))
