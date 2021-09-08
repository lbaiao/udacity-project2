import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(messages_filepath, categories_filepath):
    """Properly loads and preprocess the data.

    Parameters
    ---
        messages_filepath: str
            Messages csv file path.
        categories_filepath: str
            Categories csv file path.

    Returns
    ---
    df: pandas.DataFrame
        Dataframe containing the data, properly processed.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop(columns='original')
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(right=categories, left=messages, how='inner', on='id')
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', -1, True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df
    

def clean_data(df):
    """Remove duplicates, NaN values and messages that are
    too long.

    Parameters
    ---
    df: pandas.DataFrame
        Dataframe with the preprocessed data.

    Returns
    ---
    df: pandas.DataFrame
        Dataframe without duplicates and NaN values.
    """
    # drop duplicates
    df = df.drop_duplicates()
    # since i do not know the meaning of label "2", i will drop these rows, since they are just a few.  # noqa
    df = df[df['related'] < 2]
    # drop null values
    df = df.dropna()
    # drop long messages
    messages_lens = np.array([len(i.split(' ')) for i in df['message']])
    x = np.linspace(1, 200, 100)
    cdf = np.array([np.mean(messages_lens < i) for i in x])
    len_lim = np.ceil(x[cdf > .99][0])
    df = df.iloc[messages_lens <= len_lim]
    return df


def save_data(df, database_filename):
    """Loads the preprocessed data into a SQLite database.

    Parameters
    ---
    df: pandas.DataFrame
        Dataframe with the data the will be loaded into the database.
    database_filename: str
        SQLite database filename.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False)


def main():
    """Executes the ETL pipeline.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
